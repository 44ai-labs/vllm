# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict, deque
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar
from vllm.v1.utils import ConstantList

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5])
def test_stop_by_max_tokens(max_tokens: int):
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=2, max_tokens=max_tokens)
    req0, req1 = requests

    expected_total_num_scheduled_tokens = 0
    sched_outputs: deque[SchedulerOutput] = deque()
    scheduler.add_request(req0)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req0.num_prompt_tokens + max_tokens - 1

    scheduler.add_request(req1)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req1.num_prompt_tokens + max_tokens - 1

    total_num_scheduled_tokens = 0
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        total_num_scheduled_tokens += sched_output.total_num_scheduled_tokens
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req0.num_output_tokens == max_tokens
    assert req1.num_output_tokens == max_tokens
    # Ensure we aren't scheduling more tokens than necessary.
    assert total_num_scheduled_tokens == expected_total_num_scheduled_tokens


def test_async_per_request_spec_decode_opt_out():
    """Per-request MTP opt-out must hold under async scheduling.

    Regression for the async-scheduling gap: the sync gate lives in
    ``update_draft_token_ids``, which the async path never calls.
    ``AsyncScheduler._update_after_schedule`` must clear ``spec_token_ids`` for
    opted-out requests so the next ``schedule()`` reserves no speculative slots for
    them, while opted-in requests keep their placeholders.
    """
    scheduler = create_scheduler(
        async_scheduling=True, num_speculative_tokens=3, speculative_method="ngram_gpu"
    )

    # Separate create_requests calls so each request owns its SamplingParams.
    opted_in = create_requests(num_requests=1, max_tokens=10, req_ids=["in"])[0]
    opted_out = create_requests(num_requests=1, max_tokens=10, req_ids=["out"])[0]
    opted_in.sampling_params.enable_speculative_decoding = True
    opted_out.sampling_params.enable_speculative_decoding = False
    scheduler.add_request(opted_in)
    scheduler.add_request(opted_out)

    # Prefill (no spec scheduled yet), then the first decode step. The prefill step's
    # _update_after_schedule sets spec_token_ids per opt-in; the next schedule() acts
    # on it. Only the prefill output is fed back, so no spec-output mocking is needed.
    sched0 = scheduler.schedule()
    scheduler.update_from_output(sched0, _make_model_runner_output(sched0))
    sched1 = scheduler.schedule()

    # The opt-in gate is reflected directly in spec_token_ids ...
    assert opted_in.spec_token_ids == [-1, -1, -1]
    assert opted_out.spec_token_ids == []
    # ... and in what the next step actually schedules.
    assert len(sched1.scheduled_spec_decode_tokens.get("in", [])) == 3
    assert len(sched1.scheduled_spec_decode_tokens.get("out", [])) == 0


def test_abort():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_preempt():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_prefix_caching_for_prefill_dedup():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=3,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )

    # Two requests with the same prompt.
    req0 = requests.pop(0)
    req1 = requests.pop(0)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)
    # Make sure prefix caching de-duplicates the prompts in the same step,
    # so all the blocks except the last are shared between the two requests.
    assert len(sched_output.num_scheduled_tokens) == 2
    assert sched_output.num_scheduled_tokens[req0.request_id] == num_prompt_tokens
    assert (
        sched_output.num_scheduled_tokens[req1.request_id]
        == num_prompt_tokens % BLOCK_SIZE
    )

    sched_outputs.append(scheduler.schedule())
    while sched_outputs:
        added_req = None
        if requests:
            added_req = requests.pop(0)
            scheduler.add_request(added_req)
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
            if added_req:
                assert (
                    sched_output.num_scheduled_tokens[added_req.request_id]
                    == num_prompt_tokens % BLOCK_SIZE
                )

    assert scheduler.get_num_unfinished_requests() == 0


def test_prefix_caching_for_multi_turn():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    num_output_tokens = 200
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )

    for req in requests:
        scheduler.add_request(req)
    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    # Process the requests.
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
    assert scheduler.get_num_unfinished_requests() == 0

    # Create next-turn requests whose prompts are the full output of the
    # previous turn.
    next_turn_requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens + num_output_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )
    for i, req in enumerate(next_turn_requests):
        req.prompt_token_ids = requests[i].prompt_token_ids + list(
            requests[i].output_token_ids
        )
        req._all_token_ids = req.prompt_token_ids.copy()
        req.all_token_ids = ConstantList(req._all_token_ids)
        req.block_hashes = []
        req.update_block_hashes()

    # Schedule the next-turn requests.
    for req in next_turn_requests:
        scheduler.add_request(req)
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)

    # Make sure the next-turn requests get prefix cache hit by the previous
    # requests.
    for req in next_turn_requests:
        assert sched_output.num_scheduled_tokens[req.request_id] == (
            req.num_prompt_tokens % BLOCK_SIZE
        )


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(AsyncScheduler)
    request = create_requests(num_requests=1, num_tokens=1)[0]
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    request.structured_output_request.grammar.accept_tokens.return_value = False
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.structured_output_manager.trim_reasoning_for_advance.side_effect = (
        lambda request, new_token_ids: new_token_ids
    )
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.grammar_compile_error_reqs = set()
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req, delay_free_blocks=False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None, None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(output, model_runner_output)

    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running


class PipelinedEngine:
    """Drive a real AsyncScheduler like EngineCore.step_with_batch_queue:
    schedule until the batch queue is full, then process the oldest step's
    output. Async PP runs pp_size+1 concurrent batches, so up to pp_size
    steps are in flight at each schedule() call -- the window in which
    preemption must handle output that has not yet returned. (Single-GPU e2e
    tests can never create this window: at PP=1, exactly one step is in
    flight and it is processed before a preempted request can resume.)

    The model runner is emulated with the V2 runner's own bookkeeping, from
    only what the scheduler serializes to it: slots flushed on
    preempted_req_ids, resumed requests re-added from the NewRequestData
    snapshot, sampling when a step reaches the end of the runner's own view
    of the sequence. This makes preemption races observable: a stale token
    delivered after a resume is scheduled extends the scheduler's sequence
    but not the runner's.

    Every sample emits a globally unique token tagged with its sampled
    position, so tests can assert exact delivery.
    """

    def __init__(
        self,
        scheduler: AsyncScheduler,
        queue_size: int,
        accept_drafts: Callable[[int, str, int], int] | None = None,
    ):
        self.scheduler = scheduler
        self.queue_size = queue_size
        self.accept_drafts = accept_drafts
        # In-flight steps: (scheduler_output, new_reqs snapshot) in FIFO order.
        self.queue: deque[tuple[SchedulerOutput, list[tuple[str, int, int]]]] = deque()
        # Runner-side request state: req_id -> [seq_len, num_computed] as the
        # runner sees them (its own sampled tokens, not the scheduler's).
        self.runner_view: dict[str, list[int]] = {}
        # All tokens the fake runner ever sampled, per request, in order.
        self.emitted: dict[str, list[int]] = defaultdict(list)
        # Sequence position each (globally unique) token was sampled for.
        self.emitted_position: dict[int, int] = {}
        self.step_idx = 0
        self._next_token = 1000

    def _schedule(self) -> bool:
        scheduler_output = self.scheduler.schedule()
        self.step_idx += 1
        # Snapshot what NewRequestData serializes at schedule time (both new
        # and resumed requests for the V2 runner).
        new_reqs = [
            (r.req_id, len(r.prefill_token_ids), r.num_computed_tokens)
            for r in scheduler_output.scheduled_new_reqs
        ]
        # Enqueue empty steps too (the engine executes them), so the runner
        # still observes their preempted/finished request ids in step order.
        self.queue.appendleft((scheduler_output, new_reqs))
        return True

    def _process_oldest_step(self) -> None:
        scheduler_output, new_reqs = self.queue.pop()
        # Worker-side state updates, in step order: flush preempted/finished
        # slots, then (re-)add new/resumed requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            self.runner_view.pop(req_id, None)
        for req_id in scheduler_output.finished_req_ids or ():
            self.runner_view.pop(req_id, None)
        for req_id, seq_len, num_computed in new_reqs:
            self.runner_view[req_id] = [seq_len, num_computed]

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        sampled_token_ids: list[list[int]] = []
        for req_id in req_ids:
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            view = self.runner_view.get(req_id)
            if view is None:
                # Slot already flushed (request finished/aborted mid-flight).
                sampled_token_ids.append([])
                continue
            seq_len, num_computed = view
            end = num_computed + num_scheduled
            if end < seq_len:
                # Partial prefill by the runner's own bookkeeping: no sample.
                view[1] = end
                sampled_token_ids.append([])
                continue
            drafts = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            num_accepted = (
                min(self.accept_drafts(self.step_idx, req_id, len(drafts)), len(drafts))
                if drafts and self.accept_drafts
                else 0
            )
            num_rejected = len(drafts) - num_accepted
            tokens = list(range(self._next_token, self._next_token + 1 + num_accepted))
            self._next_token += 1 + num_accepted
            self.emitted[req_id].extend(tokens)
            sampled_token_ids.append(tokens)
            # Rejected drafts roll back computed; the sampled tokens extend
            # the runner's sequence.
            view[1] = end - num_rejected
            view[0] = view[1] + 1
            for offset, token in enumerate(tokens):
                self.emitted_position[token] = view[0] - len(tokens) + offset
        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        self.scheduler.update_from_output(scheduler_output, model_runner_output)

    def run(
        self,
        max_steps: int = 2000,
        before_step: Callable[[int, "PipelinedEngine"], None] | None = None,
    ) -> None:
        for i in range(max_steps):
            if not self.scheduler.has_requests() and not self.queue:
                return
            if before_step is not None:
                before_step(i, self)
            scheduled = (
                self.scheduler.has_requests()
                and len(self.queue) < self.queue_size
                and self._schedule()
            )
            if scheduled and len(self.queue) < self.queue_size:
                # Queue not yet full: the engine returns without blocking.
                continue
            if self.queue:
                self._process_oldest_step()
        raise AssertionError("engine loop did not converge")


def _create_async_pp_scheduler(
    num_spec: int, pp_size: int = 3, num_blocks: int = 5
) -> AsyncScheduler:
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec or None,
        speculative_method="ngram_gpu" if num_spec else None,
        use_v2_model_runner=True,
        num_blocks=num_blocks,
        block_size=16,
        max_num_batched_tokens=512,
    )
    # Emulate PP at the scheduler level; constructing with
    # pipeline_parallel_size>1 requires that many visible GPUs. Drive with
    # queue_size=pp_size+1 (V2 async PP runs pp_size+1 concurrent batches).
    scheduler.pp_size = pp_size
    scheduler.use_pp = pp_size > 1
    return scheduler


def _assert_ordered_subset(delivered: list[int], emitted: list[int]) -> None:
    """Delivered tokens must be an order-preserving subset of the emitted
    tokens with no duplicates (tokens are globally unique)."""
    it = iter(emitted)
    for token in delivered:
        assert token in it, f"token {token} delivered out of order or twice"


def _assert_positions_consistent(req, engine: PipelinedEngine) -> None:
    """The i-th delivered output token must be one the runner sampled for
    exactly sequence position prompt_len + i: catches a preempted request's
    stale output landing on a position the resumed request resampled (or
    vice versa), which token-stream equality alone cannot see."""
    for i, token in enumerate(req.output_token_ids):
        expected = req.num_prompt_tokens + i
        actual = engine.emitted_position[token]
        assert actual == expected, (
            f"output {i} of {req.request_id}: token sampled for position "
            f"{actual}, delivered as position {expected}"
        )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_kv_pressure_preemption_with_inflight_output(num_spec: int):
    """KV-pressure preemption of requests with in-flight async output.

    PP=3 + async scheduling (batch queue of 4), a block pool small enough
    that decodes contend and preempt mid-flight, and staggered arrivals so
    the batch queue actually pipelines. A preempted request's in-flight steps
    still return: their tokens must be delivered exactly once, their stale
    spec-rejection counts must not corrupt the rolled-back counters, and the
    resume must not resample a position that output later delivers.

    Regression for the num_output_placeholders underflow EngineCore crash:
    with the fix reverted, the num_spec=3 variant fails with exactly
    ``assert request.num_output_placeholders >= 0`` when a stale spec output
    returns after the preempted request was resumed and sampled.
    """
    max_tokens = 24
    scheduler = _create_async_pp_scheduler(num_spec)
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=max_tokens, ignore_eos=True
    )
    pending = list(requests)
    for _ in range(2):
        scheduler.add_request(pending.pop(0))

    # Observe that the scenario under test actually occurs.
    preempts_with_inflight_output = 0
    orig_preempt = scheduler._preempt_request

    def counting_preempt(request, timestamp, **kwargs):
        nonlocal preempts_with_inflight_output
        if request.num_in_flight_tokens > 0:
            preempts_with_inflight_output += 1
        return orig_preempt(request, timestamp, **kwargs)

    scheduler._preempt_request = counting_preempt

    def add_requests(step: int, engine: PipelinedEngine):
        if pending:
            scheduler.add_request(pending.pop(0))

    engine = PipelinedEngine(
        scheduler,
        queue_size=4,
        # Deterministically vary spec acceptance so stale outputs carry
        # nonzero rejection counts.
        accept_drafts=lambda step, req_id, n: (step + int(req_id)) % (n + 1),
    )
    engine.run(before_step=add_requests)

    assert preempts_with_inflight_output > 0, (
        "test did not exercise preemption with in-flight output"
    )
    for req in requests:
        assert req.is_finished()
        assert req.num_output_tokens == max_tokens
        # Lossless: delivered tokens are exactly the sampled tokens, in order
        # (the excluded tail was emitted after the request finished).
        emitted = engine.emitted[req.request_id]
        assert list(req.output_token_ids) == emitted[:max_tokens]
        _assert_positions_consistent(req, engine)


@pytest.mark.parametrize("pp_size", [1, 3])
def test_reset_prefix_cache_with_inflight_output_under_kv_pressure(pp_size: int):
    """reset_prefix_cache(reset_running_requests=True) resumes requests in
    the same step it preempts them, so in-flight output must be dropped (the
    resume resamples those positions).

    pp_size=1: regression for the frame-based discard this fix replaces,
    which with spec decode drained one *token* count per output frame and
    over-discarded, corrupting the fresh frames after the resume.
    pp_size=3: back-to-back resets, so the second re-preempts requests whose
    dropped stale share is still in flight -- it must be recorded once (not
    accumulated) and stay dropped.
    """
    max_tokens = 24
    scheduler = _create_async_pp_scheduler(num_spec=3, pp_size=pp_size)
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=max_tokens, ignore_eos=True
    )
    pending = list(requests)
    for _ in range(2):
        scheduler.add_request(pending.pop(0))

    # Observe re-preemptions with an undrained stale share (the
    # double-count hazard).
    repreempts_with_stale = 0
    orig_preempt = scheduler._preempt_request

    def counting_preempt(request, timestamp, **kwargs):
        nonlocal repreempts_with_stale
        if getattr(request, "num_stale_output_tokens", 0) > 0:
            repreempts_with_stale += 1
        return orig_preempt(request, timestamp, **kwargs)

    scheduler._preempt_request = counting_preempt

    resets = 0
    reset_steps = {6, 14} if pp_size == 1 else {6, 7, 18, 19}

    def before_step(step: int, engine: PipelinedEngine):
        nonlocal resets
        if pending:
            scheduler.add_request(pending.pop(0))
        if step in reset_steps and (engine.queue or scheduler.running):
            scheduler.reset_prefix_cache(reset_running_requests=True)
            resets += 1

    engine = PipelinedEngine(
        scheduler,
        queue_size=pp_size + 1,
        accept_drafts=lambda step, req_id, n: (step + int(req_id)) % (n + 1),
    )
    engine.run(before_step=before_step)

    assert resets > 0, "test did not exercise reset_prefix_cache"
    if pp_size > 1:
        # The re-preempt-while-stale-pending window needs pipeline depth.
        assert repreempts_with_stale > 0, (
            "test did not exercise re-preemption with an undrained stale share"
        )
    for req in requests:
        assert req.is_finished()
        assert req.num_output_tokens == max_tokens
        # Dropped tokens are never delivered; order must be preserved with
        # no duplicates.
        _assert_ordered_subset(
            list(req.output_token_ids), engine.emitted[req.request_id]
        )
        _assert_positions_consistent(req, engine)
        # All stale shares fully drained by the end.
        assert getattr(req, "num_stale_output_tokens", 0) == 0


def test_requires_kv_delivery_defaults_to_producer_role():
    # No connector: nothing is handed off, so keep the lossless deliver-stale
    # path on preemption.
    assert create_scheduler(async_scheduling=True).requires_kv_delivery is False
    # Only a producer hands KV off when a request completes.
    for role, expected in (
        ("kv_producer", True),
        ("kv_both", True),
        ("kv_consumer", False),
    ):
        scheduler = create_scheduler(
            async_scheduling=True, use_kv_connector=True, kv_role=role
        )
        assert scheduler.requires_kv_delivery is expected, role


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
def test_kv_pressure_preempt_mid_handoff(kv_role: str):
    """P/D race: a request is KV-pressure preempted while the output of its
    final prefill chunk -- the hand-off token that would finish it -- is in
    flight.

    On a producer, that output must be dropped so the request recomputes;
    delivering it would finish the request and hand off blocks the preemption
    already freed, so the consumer pulls garbage. A consumer hands nothing off,
    so it keeps the lossless deliver-stale path.
    """
    is_producer = kv_role == "kv_producer"
    scheduler = create_scheduler(
        async_scheduling=True,
        use_kv_connector=True,
        kv_role=kv_role,
        num_blocks=5,
        block_size=16,
        max_num_batched_tokens=512,
    )
    assert scheduler.requires_kv_delivery is is_producer

    # 32-token prompts fill 2 blocks each, exhausting the usable pool, so the
    # next decode allocation preempts the tail of the running queue (the handoff
    # request) while its prefill output is still in flight.
    decoder = create_requests(
        num_requests=1, num_tokens=32, max_tokens=8, req_ids=["decoder"]
    )[0]
    handoff = create_requests(
        num_requests=1, num_tokens=32, max_tokens=1, req_ids=["handoff"]
    )[0]
    scheduler.add_request(decoder)
    scheduler.add_request(handoff)
    sched_output = scheduler.schedule()
    assert handoff.status == RequestStatus.RUNNING
    assert handoff.num_output_placeholders == 1

    scheduler.schedule()
    assert handoff.status == RequestStatus.PREEMPTED
    assert handoff.num_stale_output_tokens == handoff.num_prompt_tokens
    assert handoff.drop_stale_output is is_producer

    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))

    assert handoff.num_stale_output_tokens == 0
    if is_producer:
        # Dropped: recomputed from the waiting queue, so the hand-off happens
        # against real KV.
        assert not handoff.is_finished()
        assert handoff.status == RequestStatus.PREEMPTED
        assert handoff.num_output_tokens == 0
        assert handoff.request_id in scheduler.requests
    else:
        assert handoff.is_finished()
        assert handoff.num_output_tokens == 1


def _jd_model_output(req_id: str, token: int) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[token]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_async_jump_forward_discards_inflight_sample():
    """Jump-forward under async scheduling must discard the in-flight sample.

    When ff tokens are emitted, the next step is already in flight: its
    forward predates the jump, so its sample is conditioned on a pre-jump
    context while the deferred mask and the committed ordering are post-jump.
    The scheduler must release the in-flight placeholders at emission time
    (so the next schedule() lays out the ff positions without dead slots),
    drop the in-flight sample when it arrives, and re-decode the position
    after the ff tokens are processed.
    """
    scheduler = create_scheduler(async_scheduling=True, enable_jump_decoding=True)
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)

    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.sampling_params.structured_outputs = Mock(enable_jump_decoding=True)
    grammar = Mock(spec=StructuredOutputGrammar)
    grammar.accept_tokens = Mock(return_value=True)
    grammar.is_terminated = Mock(return_value=False)
    grammar.advance_ff_tokens = Mock(return_value=[100, 101, 102])
    req.structured_output_request = Mock(
        grammar=grammar, reasoning_ended=None, reasoning_end_token_index=None
    )
    scheduler.add_request(req)

    # Step N (prefill + first sample) and step N+1 are scheduled before
    # step N's output is processed -- the async in-flight window.
    sched_n = scheduler.schedule()
    sched_n1 = scheduler.schedule()
    assert req.num_output_placeholders == 2

    # Step N's output commits token 7; the grammar then jumps 3 tokens.
    # The in-flight step N+1 predates the jump: its placeholder must be
    # released and its (future) sample marked for discard.
    scheduler.update_from_output(sched_n, _jd_model_output(req.request_id, 7))
    assert list(req.output_token_ids) == [7, 100, 101, 102]
    assert scheduler.pending_ff_tokens[req.request_id] == [100, 101, 102]
    assert req.jd_discard_pending == 1
    assert req.num_output_placeholders == 0

    # Step N+1's sample (9) arrives: dropped, not committed, and the
    # grammar is never advanced over it.
    scheduler.update_from_output(sched_n1, _jd_model_output(req.request_id, 9))
    assert list(req.output_token_ids) == [7, 100, 101, 102]
    assert req.jd_discard_pending == 0
    grammar.accept_tokens.assert_called_once_with(req.request_id, [7])

    # The next schedule() lays out EXACTLY the 3 ff tokens -- no dead slot
    # for the discarded sample (the broken layout would schedule 4).
    grammar.advance_ff_tokens = Mock(return_value=[])
    sched_n2 = scheduler.schedule()
    assert sched_n2.num_scheduled_tokens[req.request_id] == 3
    assert sched_n2.jump_forward_tokens == {req.request_id: [100, 101, 102]}
    # ... and reserves one fresh placeholder for the re-decoded position.
    assert req.num_output_placeholders == 1


def test_async_jump_forward_no_inflight_no_discard():
    """With no step in flight at ff-emission time there is nothing to
    discard: the placeholders the request accumulated are its own step's,
    already consumed by the commit."""
    scheduler = create_scheduler(async_scheduling=True, enable_jump_decoding=True)
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)

    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.sampling_params.structured_outputs = Mock(enable_jump_decoding=True)
    grammar = Mock(spec=StructuredOutputGrammar)
    grammar.accept_tokens = Mock(return_value=True)
    grammar.is_terminated = Mock(return_value=False)
    grammar.advance_ff_tokens = Mock(return_value=[100])
    req.structured_output_request = Mock(
        grammar=grammar, reasoning_ended=None, reasoning_end_token_index=None
    )
    scheduler.add_request(req)

    # Only step N scheduled: its own placeholder is consumed by the commit,
    # leaving none in flight when the jump is emitted.
    sched_n = scheduler.schedule()
    assert req.num_output_placeholders == 1

    scheduler.update_from_output(sched_n, _jd_model_output(req.request_id, 7))
    assert list(req.output_token_ids) == [7, 100]
    assert req.jd_discard_pending == 0
    assert req.num_output_placeholders == 0


def test_async_jump_forward_discard_rolls_back_dead_spec_positions():
    """With MTP, the in-flight step's chunk also covers K draft positions --
    exactly where the ff tokens get committed. The discard must mark those
    positions uncomputed (their KV belongs to dead drafts) and drop the dead
    spec placeholders, so the next schedule() re-runs the FULL ff span
    through the target instead of just its tail."""
    scheduler = create_scheduler(async_scheduling=True, enable_jump_decoding=True)
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)

    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.sampling_params.structured_outputs = Mock(enable_jump_decoding=True)
    grammar = Mock(spec=StructuredOutputGrammar)
    grammar.accept_tokens = Mock(return_value=True)
    grammar.is_terminated = Mock(return_value=False)
    grammar.advance_ff_tokens = Mock(return_value=[100, 101, 102])
    req.structured_output_request = Mock(
        grammar=grammar, reasoning_ended=None, reasoning_end_token_index=None
    )
    scheduler.add_request(req)

    sched_n = scheduler.schedule()
    # Simulate the MTP in-flight step: 1 sample + 4 draft lanes scheduled,
    # with the draft positions already counted as computed.
    req.spec_token_ids = [-1, -1, -1, -1]
    req.num_output_placeholders += 1 + 4
    req.num_computed_tokens += 1 + 4

    scheduler.update_from_output(sched_n, _jd_model_output(req.request_id, 7))
    assert list(req.output_token_ids) == [7, 100, 101, 102]
    assert req.jd_discard_pending == 1
    assert req.num_output_placeholders == 0
    assert req.spec_token_ids == []
    # All 3 ff positions are uncomputed again: the dead draft KV there must
    # be overwritten by the real tokens.
    assert req.num_computed_tokens == req.num_tokens - 3

    grammar.advance_ff_tokens = Mock(return_value=[])
    sched_n1 = scheduler.schedule()
    assert sched_n1.num_scheduled_tokens[req.request_id] == 3
    assert sched_n1.jump_forward_tokens == {req.request_id: [100, 101, 102]}


def test_async_jd_discard_skips_spec_rejection_accounting():
    """The discarded step's spec-rejection accounting must not run: its
    placeholders and num_computed were already settled at ff-emission time,
    and a second num_computed subtraction would desync the position math."""
    scheduler = create_scheduler(async_scheduling=True, enable_jump_decoding=True)
    scheduler.structured_output_manager.should_advance = Mock(return_value=True)

    req = create_requests(num_requests=1, max_tokens=20)[0]
    req.sampling_params.structured_outputs = Mock(enable_jump_decoding=True)
    req.structured_output_request = Mock(
        grammar=Mock(spec=StructuredOutputGrammar),
        reasoning_ended=None,
        reasoning_end_token_index=None,
    )
    scheduler.add_request(req)
    scheduler.schedule()

    req.jd_discard_pending = 1
    nc_before = req.num_computed_tokens
    dead = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req.request_id: 5},
        total_num_scheduled_tokens=5,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={req.request_id: [-1, -1, -1, -1]},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    scheduler.update_from_output(dead, _jd_model_output(req.request_id, 9))
    # 1 sampled token of 4 drafts would normally subtract 3 rejected from
    # num_computed -- must NOT happen for a discarded step; the sample
    # itself is dropped without being committed.
    assert req.num_computed_tokens == nc_before
    assert req.jd_discard_pending == 0
    assert list(req.output_token_ids) == []

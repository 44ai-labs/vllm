# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import deque
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
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
    scheduler = create_scheduler(async_scheduling=True, num_speculative_tokens=3)

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
    request.structured_output_request.grammar = Mock()
    request.structured_output_request.grammar.accept_tokens.return_value = False
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req, delay_free_blocks=False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None

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
    grammar = Mock()
    grammar.accept_tokens = Mock(return_value=True)
    grammar.is_terminated = Mock(return_value=False)
    grammar.advance_ff_tokens = Mock(return_value=[100, 101, 102])
    req.structured_output_request = Mock(grammar=grammar, reasoning_ended=None)
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
    grammar = Mock()
    grammar.accept_tokens = Mock(return_value=True)
    grammar.is_terminated = Mock(return_value=False)
    grammar.advance_ff_tokens = Mock(return_value=[100])
    req.structured_output_request = Mock(grammar=grammar, reasoning_ended=None)
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
    grammar = Mock()
    grammar.accept_tokens = Mock(return_value=True)
    grammar.is_terminated = Mock(return_value=False)
    grammar.advance_ff_tokens = Mock(return_value=[100, 101, 102])
    req.structured_output_request = Mock(grammar=grammar, reasoning_ended=None)
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
    req.structured_output_request = Mock(grammar=Mock(), reasoning_ended=None)
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import (
    Scheduler,
    _assert_jd_cache_safe,
    _request_opts_into_spec_decode,
)
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            # Per-request MTP opt-out: opted-out requests get no spec placeholders,
            # so the next schedule() reserves no speculative slots for them. The sync
            # path gates this in Scheduler.update_draft_token_ids, which async never
            # calls; mirroring it here (before the next schedule()) keeps
            # num_scheduled_tokens and KV allocation self-consistent.
            if _request_opts_into_spec_decode(request):
                request.spec_token_ids = self._spec_token_placeholders
            else:
                request.spec_token_ids = []

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        if request.jd_discard_pending > 0:
            # Jump-forward tokens were emitted while this step was in
            # flight: its sample is conditioned on a pre-jump context (the
            # ff tokens were not part of its forward pass). Drop it; the
            # position is re-decoded after the ff tokens are processed.
            # num_output_placeholders was already released at ff-emission
            # time, so no decrement here.
            request.jd_discard_pending -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            num_to_cache = request.num_computed_tokens - request.num_output_placeholders
            # jd_guaranteed_prefill: while a [D, S_pred] step is in flight, S's KV
            # is computed (so it is counted in num_computed_tokens) but S is neither
            # committed (num_tokens) nor a sampled placeholder -- it is appended to
            # the output only later in update_from_output, after the S_pred ==
            # S_actual verify. Exclude that span from the cache boundary so a span a
            # mismatch will discard is never hashed into the prefix cache before the
            # commit. pending_prefill_pred is set at schedule and cleared at that
            # verify, so it is non-empty exactly for this in-flight window (and only
            # under async, where guaranteed prefill runs); S's blocks are cached on
            # the next step, once num_tokens has caught up.
            if request.pending_prefill_pred:
                num_to_cache -= len(request.pending_prefill_pred)
            if envs.VLLM_JD_DEBUG_ASSERTS:
                _assert_jd_cache_safe(request, num_to_cache)
            self.kv_cache_manager.cache_blocks(request, num_to_cache)
        return new_token_ids, stopped

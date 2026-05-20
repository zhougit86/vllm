# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                if (
                    request.num_output_placeholders > 0
                    or bool(request.spec_token_ids)
                    or not request.allow_async_spec_reuse
                ):
                    logger.warning(
                        "[DEBUG-spec-budget] async_after_schedule_skip_prefill "
                        "req_id=%s num_scheduled_tokens=%d num_computed_tokens=%d "
                        "num_tokens=%d num_output_placeholders=%d spec_len=%d "
                        "allow_async_spec_reuse=%s",
                        req_id,
                        scheduler_output.num_scheduled_tokens[req_id],
                        request.num_computed_tokens,
                        request.num_tokens,
                        request.num_output_placeholders,
                        len(request.spec_token_ids),
                        request.allow_async_spec_reuse,
                    )
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            old_num_output_placeholders = request.num_output_placeholders
            old_spec_len = len(request.spec_token_ids)
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders
            logger.warning(
                "[DEBUG-spec-budget] async_after_schedule_seed req_id=%s "
                "num_scheduled_tokens=%d cur_num_spec_tokens=%d "
                "num_output_placeholders_before=%d num_output_placeholders_after=%d "
                "spec_len_before=%d spec_len_after=%d allow_async_spec_reuse=%s",
                req_id,
                scheduler_output.num_scheduled_tokens[req_id],
                cur_num_spec_tokens,
                old_num_output_placeholders,
                request.num_output_placeholders,
                old_spec_len,
                len(request.spec_token_ids),
                request.allow_async_spec_reuse,
            )

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        old_num_output_placeholders = request.num_output_placeholders
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0
        if old_num_output_placeholders > 0 or new_token_ids:
            logger.warning(
                "[DEBUG-spec-budget] async_update_output req_id=%s "
                "new_token_count=%d num_output_placeholders_before=%d "
                "num_output_placeholders_after=%d num_computed_tokens=%d "
                "num_tokens=%d status_before_update=%s stopped=%s",
                request.request_id,
                len(new_token_ids),
                old_num_output_placeholders,
                request.num_output_placeholders,
                request.num_computed_tokens,
                request.num_tokens,
                status_before_update,
                stopped,
            )

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped

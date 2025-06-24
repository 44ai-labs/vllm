# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from multiprocessing import Manager
from unittest.mock import patch

from vllm import LLM
from vllm.v1.core.sched.scheduler import Scheduler

# ---------- constants ----------
BIG_PROMPT_TOKENS = 800  # 3×256 + 32 remainder
STEP_BUDGET = 4096  # plenty of room per step

MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = "Hello my name is Robert and I"


# uv pip install pytest-mock
def test_deterministic_chunking(mocker):
    """Verify the 256 / remainder pattern during prefill is deterministic."""
    with (Manager() as manager
          ):  # as there is a fork when launching LLM so we need a manager
        captured = manager.list()

        real_schedule = Scheduler.schedule

        def recorder(self, *args, **kwargs):
            result = real_schedule(self, *args, **kwargs)
            print("Captured result:", result)
            print(f"captured: {id(captured)}")
            captured.append(result)
            return result

        def scheduler_init(*args, **kwargs):
            print("CALLING SCHEDULER INIT")
            new_scheduler_cls = Scheduler
            new_scheduler_cls.schedule = recorder
            return new_scheduler_cls

        with patch("vllm.v1.engine.core.resolve_obj_by_qualname",
                   new=scheduler_init):
            llm = LLM(
                MODEL,
                enforce_eager=True,
                enable_prefix_caching=True,
                long_prefill_token_threshold=2,
                max_num_batched_tokens=6,
                max_num_seqs=3,
                block_size=16,
                gpu_memory_utilization=0.3,
            )

            _outputs = llm.generate([PROMPT])

            for capture in captured:
                print(f"{capture.num_scheduled_tokens}")

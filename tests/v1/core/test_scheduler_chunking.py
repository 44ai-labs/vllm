# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from multiprocessing import Manager
from unittest.mock import patch

from vllm import LLM, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler

MODEL = "HandH1998/QQQ-Llama-3-8b-g128"  # "44ai/gemma-3-4b-it-qat-qqq"
PROMPT = """Once upon a forgotten shoreline, where silver kelp whispered secrets to the moonlit tide, lived a cartographer named Odessa Vane. She mapped impossibilities: fault lines of thunder, reefs of mirage, and the star-shaped archipelagos that only dreamers could reach. One mist-blue dawn, a bottle bobbed onto her horizon. Inside: a clockwork scarab and a note—“Find me before the last minute breaks,” signed with the sigil of an ink-eyed sun.

Curiosity outweighed caution. Odessa loaded her skiff, the Paperwing, whose sails were stitched from retired atlases, and followed the scarab’s whirring compass heart. It pointed not north, south, east, nor west, but inward, toward the Blue Between, a mythic hollow on every map. The sea flattened into mirrored glass; her reflection looked back with a stranger’s eyes. Waves peeled away like pages, revealing corridors of sky beneath.

There she met Captain Edrin Kestrel, rumored dead many times, who helmed the Lacuna, a galleon built of lost paragraphs. His ship’s figurehead was a quill writing water into sentences. Edrin explained: “Time is unbinding. Minutes are"""  # noqa: E501


# uv pip install pytest-mock
def test_deterministic_chunking(mocker):
    """Verify the 256 / remainder pattern during prefill is deterministic."""
    with (Manager() as manager
          ):  # as there is a fork when launching LLM so we need a manager
        captured = manager.list()

        real_schedule = Scheduler.schedule

        def recorder(self, *args, **kwargs):
            result = real_schedule(self, *args, **kwargs)
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
                dtype="half",
                # long_prefill_token_threshold=2,
                max_num_batched_tokens=10,
                max_num_seqs=4,
                # block_size=16,
                gpu_memory_utilization=0.3,
            )

            seed = 4419
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=0.95,
                seed=seed,
                max_tokens=32,  # 1024,
                logprobs=20,
            )
            _outputs = llm.generate([PROMPT] * 5,
                                    sampling_params=sampling_params)
            logprobs_storage = []
            for output in _outputs:
                print(f"Output: {output.outputs[0].text}, {output.request_id},"
                      f"{len(output.outputs[0].token_ids)}")
                logprobs_storage.append(output.outputs[0].logprobs)

            # compare to logprobs in position 0 and group by different answers
            base_logprobs = logprobs_storage[0]

            total_diffs = 0
            for i, logprobs in enumerate(logprobs_storage):
                if len(logprobs) != len(base_logprobs):
                    print(f"Output logprobs length mismatch for index {i}: "
                          f"{len(logprobs)} != {len(base_logprobs)}")
                    continue
                value_diffs = 0
                for j, logprob in enumerate(logprobs):
                    if logprob != base_logprobs[j]:
                        logprob_items = logprob.items()
                        base_logprobs[j].items()
                        for k, _v in logprob_items:
                            if k not in base_logprobs[j]:
                                value_diffs += 1
                            else:
                                base_value = base_logprobs[j][k]
                                if base_value != logprob[k]:
                                    value_diffs += 1
                if value_diffs > 0:
                    print(
                        f"Output logprobs mismatch for index {i}: {value_diffs}"
                        " mismatches found.")
                    total_diffs += 1
            if total_diffs > 0:
                print(f"Total logprobs request mismatches: {total_diffs}")

            scheduler_chunking = {}  # type: ignore
            for capture in captured:
                print(f"{capture.num_scheduled_tokens}")
                for key, value in capture.num_scheduled_tokens.items():
                    if key not in scheduler_chunking:
                        scheduler_chunking[key] = []
                    scheduler_chunking[key].append(value)

            print("Scheduler chunking patterns:")
            print(scheduler_chunking)

            if total_diffs == 0:
                print("SUCCESS All logprobs requests match across outputs.")

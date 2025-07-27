# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import io
import math
import time
from collections.abc import AsyncGenerator
from functools import cached_property
from typing import Callable, Literal, Optional, TypeVar, Union, cast

import numpy as np
from fastapi import Request

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    DeltaMessage, ErrorResponse, RequestResponseMetadata,
    TranscriptionResponse, TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse, TranslationResponse,
    TranslationResponseStreamChoice, TranslationStreamResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    SpeechToTextRequest)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.models import SupportsTranscription
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

SpeechToTextResponse = Union[TranscriptionResponse, TranslationResponse]
T = TypeVar("T", bound=SpeechToTextResponse)

logger = init_logger(__name__)


class OpenAISpeechToText(OpenAIServing):
    """Base class for speech-to-text operations like transcription and 
    translation."""

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
        task_type: Literal["transcribe", "translate"] = "transcribe",
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        self.task_type = task_type

        self.asr_config = self.model_cls.get_speech_to_text_config(
            model_config, task_type)

        self.max_audio_filesize_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

        if self.default_sampling_params:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                self.default_sampling_params)

    @cached_property
    def model_cls(self) -> type[SupportsTranscription]:
        from vllm.model_executor.model_loader import get_model_cls
        model_cls = get_model_cls(self.model_config)
        return cast(type[SupportsTranscription], model_cls)

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
    ) -> tuple[list[PromptType], float]:
        # Validate request
        language = self.model_cls.validate_language(request.language)

        if len(audio_data) / 1024**2 > self.max_audio_filesize_mb:
            raise ValueError("Maximum file size exceeded.")

        with io.BytesIO(audio_data) as bytes_:
            # NOTE resample to model SR here for efficiency. This is also a
            # pre-requisite for chunking, as it assumes Whisper SR.
            y, sr = librosa.load(bytes_, sr=self.asr_config.sample_rate)

        duration = librosa.get_duration(y=y, sr=sr)
        do_split_audio = (self.asr_config.allow_audio_chunking
                          and duration > self.asr_config.max_audio_clip_s)
        chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
        prompts = []
        for chunk in chunks:
            # The model has control over the construction, as long as it
            # returns a valid PromptType.
            prompt = self.model_cls.get_generation_prompt(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                language=language,
                task_type=self.task_type,
                request_prompt=request.prompt)
            prompts.append(prompt)
        return prompts, duration

    async def _create_speech_to_text(
        self,
        audio_data: bytes,
        request: SpeechToTextRequest,
        raw_request: Request,
        response_class: type[T],
        stream_generator_method: Callable[..., AsyncGenerator[str, None]],
    ) -> Union[T, AsyncGenerator[str, None], ErrorResponse]:
        """Base method for speech-to-text operations like transcription and 
        translation."""
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.response_format not in ['text', 'json']:
            return self.create_error_response(
                "Currently only support response_format `text` or `json`")

        request_id = f"{self.task_type}-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(request)

            if lora_request:
                return self.create_error_response(
                    "Currently do not support LoRA for "
                    f"{self.task_type.title()}.")

            prompts, duration_s = await self._preprocess_speech_to_text(
                request=request,
                audio_data=audio_data,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        list_result_generator: Optional[list[AsyncGenerator[RequestOutput,
                                                            None]]] = None
        try:
            # Unlike most decoder-only models, whisper generation length is not
            # constrained by the size of the input audio, which is mapped to a
            # fixed-size log-mel-spectogram.
            default_max_tokens = self.model_config.max_model_len
            sampling_params = request.to_sampling_params(
                default_max_tokens, self.default_sampling_params)

            self._log_inputs(
                request_id,
                # It will not display special tokens like <|startoftranscript|>
                request.prompt,
                params=sampling_params,
                lora_request=None)

            beam_search = request.use_beam_search or False
            beam_size = request.beam_size or 5
            if beam_search:
                if request.stream:
                    return self.create_error_response(
                        "Streaming is not supported for beam search.")
                # TODO: check if prefix caching is enabled, and give
                # warning when not
                # "fake" beam search only works with prefix caching enabled

                # Perform beam search with the new method
                list_result_generator = await self._perform_beam_search(
                    prompts, sampling_params, request_id, beam_size)
            else:
                list_result_generator = [
                    self.engine_client.generate(
                        prompt,
                        sampling_params,
                        request_id,
                    ) for prompt in prompts
                ]
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        if request.stream:
            return stream_generator_method(request, list_result_generator,
                                           request_id, request_metadata,
                                           duration_s)
        # Non-streaming response.
        try:
            assert list_result_generator is not None
            text = ""
            for result_generator in list_result_generator:
                async for op in result_generator:
                    text += op.outputs[0].text
            return cast(T, response_class(text=text))
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def _speech_to_text_stream_generator(
        self,
        request: SpeechToTextRequest,
        list_result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        chunk_object_type: Literal["translation.chunk", "transcription.chunk"],
        response_stream_choice_class: Union[
            type[TranscriptionResponseStreamChoice],
            type[TranslationResponseStreamChoice]],
        stream_response_class: Union[type[TranscriptionStreamResponse],
                                     type[TranslationStreamResponse]],
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model

        completion_tokens = 0
        num_prompt_tokens = 0

        include_usage = request.stream_include_usage \
            if request.stream_include_usage else False
        include_continuous_usage = request.stream_continuous_usage_stats\
            if include_usage and request.stream_continuous_usage_stats\
            else False

        try:
            for result_generator in list_result_generator:
                async for res in result_generator:
                    # On first result.
                    if res.prompt_token_ids is not None:
                        num_prompt_tokens = len(res.prompt_token_ids)
                        if audio_tokens := self.model_cls.get_num_audio_tokens(
                                audio_duration_s, self.asr_config,
                                self.model_config):
                            num_prompt_tokens += audio_tokens

                    # We need to do it here, because if there are exceptions in
                    # the result_generator, it needs to be sent as the FIRST
                    # response (by the try...catch).

                    # Just one output (n=1) supported.
                    assert len(res.outputs) == 1
                    output = res.outputs[0]

                    delta_message = DeltaMessage(content=output.text)
                    completion_tokens += len(output.token_ids)

                    if output.finish_reason is None:
                        # Still generating, send delta update.
                        choice_data = response_stream_choice_class(
                            delta=delta_message)
                    else:
                        # Model is finished generating.
                        choice_data = response_stream_choice_class(
                            delta=delta_message,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)

                    chunk = stream_response_class(id=request_id,
                                                  object=chunk_object_type,
                                                  created=created_time,
                                                  choices=[choice_data],
                                                  model=model_name)

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        total_tokens=num_prompt_tokens +
                                        completion_tokens)

                final_usage_chunk = stream_response_class(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage)
                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=num_prompt_tokens + completion_tokens)

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in %s stream generator.", self.task_type)
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    def _split_audio(self, audio_data: np.ndarray,
                     sample_rate: int) -> list[np.ndarray]:
        chunk_size = sample_rate * self.asr_config.max_audio_clip_s
        overlap_size = sample_rate * self.asr_config.overlap_chunk_second
        chunks = []
        i = 0
        while i < audio_data.shape[-1]:
            if i + chunk_size >= audio_data.shape[-1]:
                # handle last chunk
                chunks.append(audio_data[..., i:])
                break

            # Find the best split point in the overlap region
            search_start = i + chunk_size - overlap_size
            search_end = min(i + chunk_size, audio_data.shape[-1])
            split_point = self._find_split_point(audio_data, search_start,
                                                 search_end)

            # Extract chunk up to the split point
            chunks.append(audio_data[..., i:split_point])
            i = split_point
        return chunks

    def _find_split_point(self, wav: np.ndarray, start_idx: int,
                          end_idx: int) -> int:
        """Find the best point to split audio by 
        looking for silence or low amplitude.
        Args:
            wav: Audio tensor [1, T]
            start_idx: Start index of search region
            end_idx: End index of search region
        Returns:
            Index of best splitting point
        """
        segment = wav[start_idx:end_idx]

        # Calculate RMS energy in small windows
        min_energy = math.inf
        quietest_idx = 0
        min_energy_window = self.asr_config.min_energy_split_window_size
        assert min_energy_window is not None
        for i in range(0, len(segment) - min_energy_window, min_energy_window):
            window = segment[i:i + min_energy_window]
            energy = (window**2).mean()**0.5
            if energy < min_energy:
                quietest_idx = i + start_idx
                min_energy = energy
        return quietest_idx

    async def _perform_beam_search(
            self, prompts: list[PromptType], sampling_params: SamplingParams,
            request_id: str,
            beam_size: int) -> list[AsyncGenerator[RequestOutput, None]]:
        """Perform beam search for speech-to-text generation.
        
        Args:
            prompts: List of prompts to process
            sampling_params: Sampling parameters for generation
            request_id: Request ID for tracking
            beam_size: Number of beams to maintain
            
        Returns:
            List of async generators for beam search results
        """
        # Optimization 1: Skip tokenizer loading and assume EOS token ID
        eos_token_id = 2  # Assume EOS token ID is 2
        print(f"Using assumed EOS token ID: {eos_token_id}")

        # Optimization 3: Set logprob threshold to filter out low-probability
        # candidates
        logprob_threshold = -5.0  # around 1%

        beam_results = []

        for prompt_idx, prompt in enumerate(prompts):
            # Initialize beam with empty tokens and text
            # Each beam element: (tokens, text, cumulative_logprob, is_finished)
            beams = [([], "", 0.0, False)]

            # Generate tokens one by one up to max_tokens
            max_beam_tokens = sampling_params.max_tokens
            print(f"Starting beam search for prompt {prompt_idx} "
                  f"with max tokens {max_beam_tokens}")
            print(f"prompt: {prompt}")

            for step in range(max_beam_tokens):
                if not beams or all(beam[3] for beam in beams):
                    break  # All beams finished

                new_beams = []
                print(f"Step {step + 1}/{max_beam_tokens} for prompt "
                      f"{prompt_idx}, current beams: {len(beams)}")

                # Optimization 2: Run beam requests in parallel
                generation_tasks = []
                beam_indices = []

                # Prepare generation tasks for each active beam
                for beam_idx, (beam_tokens, beam_text, beam_logprob,
                               is_finished) in enumerate(beams):
                    if is_finished:
                        new_beams.append(
                            (beam_tokens, beam_text, beam_logprob, True))
                        continue

                    # Validate prompt format
                    if (not isinstance(prompt, dict)
                            or "multi_modal_data" not in prompt
                            or "prompt_token_ids" not in prompt):
                        raise ValueError(
                            "Beam search requires prompts to be in dict "
                            "format with 'prompt_token_ids' and "
                            "'multi_modal_data'.")

                    # Create current prompt + generated tokens so far
                    current_prompt = {
                        "prompt_token_ids":
                        (prompt["prompt_token_ids"] + beam_tokens),
                        "multi_modal_data":
                        prompt.get("multi_modal_data", {})
                    }

                    # Create sampling params for getting logprobs
                    beam_sampling_params = SamplingParams(
                        max_tokens=1,  # Generate only one token
                        temperature=0.0,  # Use greedy for stability
                        top_k=beam_size * 2,  # Get more candidates
                        logprobs=beam_size * 2,  # Get logprobs for candidates
                        n=1,
                    )

                    # Create generation task
                    generation_task = self.engine_client.generate(
                        current_prompt, beam_sampling_params,
                        f"{request_id}-beam-{prompt_idx}-{step}-{beam_idx}")
                    generation_tasks.append(generation_task)
                    beam_indices.append(
                        (beam_idx, beam_tokens, beam_text, beam_logprob))

                # Run all generation tasks in parallel and collect results
                if generation_tasks:
                    await self._process_parallel_beam_tasks(
                        generation_tasks, beam_indices, new_beams,
                        eos_token_id, logprob_threshold, max_beam_tokens)

                # Select top beam_size beams based on cumulative logprob
                new_beams.sort(key=lambda x: x[2], reverse=True)
                beams = new_beams[:beam_size]

                if not beams:
                    break

            # Store the best beam for this prompt
            if beams:
                best_beam = max(beams, key=lambda x: x[2])  # Best by logprob
                print(f"Best beam for prompt {prompt_idx}: {best_beam}")
                beam_results.append(best_beam)
            else:
                beam_results.append(([], "", 0.0, True))  # Empty result

        # Generate final results using the beam search outputs
        return self._create_beam_result_generators(beam_results, prompts,
                                                   request_id)

    async def _process_parallel_beam_tasks(self, generation_tasks: list,
                                           beam_indices: list, new_beams: list,
                                           eos_token_id: Optional[int],
                                           logprob_threshold: float,
                                           max_beam_tokens: int) -> None:
        """Process parallel beam generation tasks efficiently."""

        # Process all tasks concurrently using gather for better performance
        async def process_single_task(task_idx: int, result_generator) -> None:
            """Process a single beam generation task."""
            beam_idx, beam_tokens, beam_text, beam_logprob = beam_indices[
                task_idx]

            try:
                async for result in result_generator:
                    output = result.outputs[0] if result.outputs else None
                    if not output or not output.logprobs:
                        new_beams.append(
                            (beam_tokens, beam_text, beam_logprob, True))
                        return

                    # Process logprobs efficiently and extract text from output
                    candidates_added = self._process_beam_candidates(
                        beam_tokens, beam_text, beam_logprob,
                        output.logprobs[0], eos_token_id, logprob_threshold,
                        max_beam_tokens, output.finish_reason, new_beams)

                    print(f"Beam {beam_idx}: added {candidates_added} "
                          f"candidates")
                    return

            except Exception as e:
                print(f"Error in beam {beam_idx}: {e}")
                new_beams.append((beam_tokens, beam_text, beam_logprob, True))

        # Run all tasks concurrently for better performance
        try:
            await asyncio.gather(*[
                process_single_task(idx, task)
                for idx, task in enumerate(generation_tasks)
            ],
                                 return_exceptions=True)
        except Exception as e:
            print(f"Error in parallel beam processing: {e}")
            # Fallback: mark remaining beams as finished
            for beam_idx, beam_tokens, beam_text, beam_logprob in beam_indices:
                new_beams.append((beam_tokens, beam_text, beam_logprob, True))

    def _process_beam_candidates(self, beam_tokens: list, beam_text: str,
                                 beam_logprob: float, logprobs_dict: dict,
                                 eos_token_id: Optional[int],
                                 logprob_threshold: float,
                                 max_beam_tokens: int,
                                 finish_reason: Optional[str],
                                 new_beams: list) -> int:
        """Process candidates for a single beam efficiently."""
        candidates_added = 0

        # Pre-filter candidates by logprob threshold for efficiency
        valid_candidates = [
            (token_id, logprob_data)
            for token_id, logprob_data in logprobs_dict.items()
            if logprob_data.logprob >= logprob_threshold
        ]

        if not valid_candidates:
            # No valid candidates, mark beam as finished
            new_beams.append((beam_tokens, beam_text, beam_logprob, True))
            return 0

        for token_id, logprob_data in valid_candidates:
            new_beam_tokens = beam_tokens + [token_id]
            new_beam_text = beam_text + logprob_data.decoded_token
            new_beam_logprob = beam_logprob + logprob_data.logprob

            # Determine if beam should finish
            is_beam_finished = (
                (eos_token_id is not None and token_id == eos_token_id)
                or (finish_reason is not None and finish_reason != "length")
                or (len(new_beam_tokens) >= max_beam_tokens))

            new_beams.append((new_beam_tokens, new_beam_text, new_beam_logprob,
                              is_beam_finished))
            candidates_added += 1

        return candidates_added

    def _create_beam_result_generators(
            self, beam_results: list, prompts: list[PromptType],
            request_id: str) -> list[AsyncGenerator[RequestOutput, None]]:
        """Create async generators for beam search results."""
        list_result_generator = []

        for prompt_idx, (best_tokens, best_text, best_logprob,
                         _) in enumerate(beam_results):
            # Use default arguments to capture values by value, not reference
            async def create_beam_result(tokens=best_tokens,
                                         text=best_text,
                                         logprob=best_logprob,
                                         idx=prompt_idx):
                # Use the stored text directly, no tokenizer needed
                generated_text = text if text else ""

                completion_output = CompletionOutput(
                    index=0,
                    text=generated_text,
                    token_ids=tokens,
                    cumulative_logprob=logprob,
                    logprobs=None,  # Could include detailed logprobs if needed
                    finish_reason="stop" if tokens else "length")

                request_output = RequestOutput(
                    request_id=f"{request_id}-beam-{idx}",
                    prompt=prompts[idx],
                    prompt_token_ids=[],  # Would need proper tokenization
                    prompt_logprobs=None,
                    outputs=[completion_output],
                    finished=True)

                yield request_output

            list_result_generator.append(create_beam_result())

        return list_result_generator

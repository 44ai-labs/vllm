# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for guided decoding tests
import io
import json
import time
from unittest.mock import patch

import librosa
import numpy as np
import openai
import pytest
import soundfile as sf
from openai._base_client import AsyncAPIClient

from vllm.assets.audio import AudioAsset, AudioAssets44ai

from ...utils import RemoteOpenAIServer

MISTRAL_FORMAT_ARGS = [
    "--tokenizer_mode", "mistral", "--config_format", "mistral",
    "--load_format", "mistral"
]


@pytest.fixture
def mary_had_lamb():
    path = AudioAsset('mary_had_lamb').get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def winning_call():
    path = AudioAsset('winning_call').get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.fixture
def audio_44ai():
    path = AudioAssets44ai(
        '30s_test_swiss_german_7kz53lbjqr.mp3').get_local_path()
    with open(str(path), "rb") as f:
        yield f


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["openai/whisper-large-v3-turbo", "mistralai/Voxtral-Mini-3B-2507"])
async def test_basic_audio(mary_had_lamb, model_name):
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    # Based on https://github.com/openai/openai-cookbook/blob/main/examples/Whisper_prompting_guide.ipynb.
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            temperature=0.0)
        out = json.loads(transcription)['text']
        assert "Mary had a little lamb," in out


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["mistralai/Voxtral-Mini-3B-2507"])
async def test_beam_search_transcription(audio_44ai, model_name):
    """Test beam search functionality for transcription."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        transcription_regular = await client.audio.transcriptions.create(
            model=model_name,
            file=audio_44ai,
            language="de",
            response_format="text",
            temperature=0.0,
            extra_body={"use_beam_search": False})
        out_regular = json.loads(transcription_regular)['text']
        # uncertainty_scores_regular = json.loads(
        #     transcription_regular)['uncertainty']
        step_sizes_regular = json.loads(transcription_regular)['step_sizes']

        print("Step sizes (regular):", step_sizes_regular)

        # Test with beam search enabled
        audio_44ai.seek(0)  # Reset file pointer
        transcription_beam = await client.audio.transcriptions.create(
            model=model_name,
            file=audio_44ai,
            language="de",
            response_format="text",
            temperature=0.0,
            extra_body={
                "use_beam_search": True,
                "beam_size": 5,
                "step_sizes": step_sizes_regular
            })
        out_beam = json.loads(transcription_beam)['text']
        # uncertainty_scores = json.loads(transcription_beam)['uncertainty']
        # step_sizes = json.loads(transcription_beam)['step_sizes']

        # Test without beam search for comparison

        # Both should contain the expected text
        assert "habe ich das Gefühl" in out_beam
        assert "habe ich das Gefühl" in out_regular

        print(f"Regular transcription: {out_regular}")
        print(f"Beam search transcription: {out_beam}")

        # Beam search should not fail and should produce valid output
        assert len(out_beam.strip()) > 0
        assert isinstance(out_beam, str)


@pytest.mark.asyncio
async def test_beam_search_validation(mary_had_lamb):
    """Test validation of beam search parameters."""
    model_name = "mistralai/Voxtral-Mini-3B-2507"
    server_args = ["--enforce-eager"]

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # Test that streaming with beam search is rejected
        with pytest.raises((openai.BadRequestError, Exception)):
            await client.audio.transcriptions.create(model=model_name,
                                                     file=mary_had_lamb,
                                                     language="en",
                                                     temperature=0.0,
                                                     extra_body={
                                                         "use_beam_search":
                                                         True,
                                                         "beam_size": 3,
                                                     },
                                                     stream=True)


@pytest.mark.asyncio
async def test_beam_search_parameters(mary_had_lamb):
    """Test different beam search parameter combinations."""
    model_name = "mistralai/Voxtral-Mini-3B-2507"
    server_args = ["--enforce-eager"]

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # Test with different beam sizes
        for beam_size in [1, 3, 5]:
            mary_had_lamb.seek(0)  # Reset file pointer
            transcription = await client.audio.transcriptions.create(
                model=model_name,
                file=mary_had_lamb,
                language="en",
                response_format="text",
                temperature=0.0,
                use_beam_search=True,
                beam_size=beam_size)
            out = json.loads(transcription)['text']
            assert len(out.strip()) > 0
            assert "Mary had a little lamb," in out


@pytest.mark.asyncio
async def test_bad_requests(mary_had_lamb):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # invalid language
        with pytest.raises(openai.BadRequestError):
            await client.audio.transcriptions.create(model=model_name,
                                                     file=mary_had_lamb,
                                                     language="hh",
                                                     temperature=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["openai/whisper-large-v3-turbo"])
async def test_long_audio_request(mary_had_lamb, model_name):
    server_args = ["--enforce-eager"]

    if model_name.startswith("openai"):
        return

    mary_had_lamb.seek(0)
    audio, sr = librosa.load(mary_had_lamb)
    # Add small silence after each audio for repeatability in the split process
    audio = np.pad(audio, (0, 1600))
    repeated_audio = np.tile(audio, 10)
    # Repeated audio to buffer
    buffer = io.BytesIO()
    sf.write(buffer, repeated_audio, sr, format='WAV')
    buffer.seek(0)
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=buffer,
            language="en",
            response_format="text",
            temperature=0.0)
        out = json.loads(transcription)['text']
        counts = out.count("Mary had a little lamb")
        assert counts == 10, counts


@pytest.mark.asyncio
async def test_non_asr_model(winning_call):
    # text to text model
    model_name = "JackFram/llama-68m"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res = await client.audio.transcriptions.create(model=model_name,
                                                       file=winning_call,
                                                       language="en",
                                                       temperature=0.0)
        err = res.error
        assert err["code"] == 400 and not res.text
        assert err[
            "message"] == "The model does not support Transcriptions API"


@pytest.mark.asyncio
async def test_completion_endpoints():
    # text to text model
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res = await client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }])
        err = res.error
        assert err["code"] == 400
        assert err[
            "message"] == "The model does not support Chat Completions API"

        res = await client.completions.create(model=model_name, prompt="Hello")
        err = res.error
        assert err["code"] == 400
        assert err["message"] == "The model does not support Completions API"


@pytest.mark.asyncio
async def test_streaming_response(winning_call):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    transcription = ""
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        res_no_stream = await client.audio.transcriptions.create(
            model=model_name,
            file=winning_call,
            response_format="json",
            language="en",
            temperature=0.0)
        # Unfortunately this only works when the openai client is patched
        # to use streaming mode, not exposed in the transcription api.
        original_post = AsyncAPIClient.post

        async def post_with_stream(*args, **kwargs):
            kwargs['stream'] = True
            return await original_post(*args, **kwargs)

        with patch.object(AsyncAPIClient, "post", new=post_with_stream):
            client = remote_server.get_async_client()
            res = await client.audio.transcriptions.create(
                model=model_name,
                file=winning_call,
                language="en",
                temperature=0.0,
                extra_body=dict(stream=True),
                timeout=30)
            # Reconstruct from chunks and validate
            async for chunk in res:
                # just a chunk
                text = chunk.choices[0]['delta']['content']
                transcription += text

        assert transcription == res_no_stream.text


@pytest.mark.asyncio
async def test_stream_options(winning_call):
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        original_post = AsyncAPIClient.post

        async def post_with_stream(*args, **kwargs):
            kwargs['stream'] = True
            return await original_post(*args, **kwargs)

        with patch.object(AsyncAPIClient, "post", new=post_with_stream):
            client = remote_server.get_async_client()
            res = await client.audio.transcriptions.create(
                model=model_name,
                file=winning_call,
                language="en",
                temperature=0.0,
                extra_body=dict(stream=True,
                                stream_include_usage=True,
                                stream_continuous_usage_stats=True),
                timeout=30)
            final = False
            continuous = True
            async for chunk in res:
                if not len(chunk.choices):
                    # final usage sent
                    final = True
                else:
                    continuous = continuous and hasattr(chunk, 'usage')
            assert final and continuous


@pytest.mark.asyncio
async def test_sampling_params(mary_had_lamb):
    """
    Compare sampling with params and greedy sampling to assert results
    are different when extreme sampling parameters values are picked. 
    """
    model_name = "openai/whisper-small"
    server_args = ["--enforce-eager"]
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            temperature=0.8,
            extra_body=dict(seed=42,
                            repetition_penalty=1.9,
                            top_k=12,
                            top_p=0.4,
                            min_p=0.5,
                            frequency_penalty=1.8,
                            presence_penalty=2.0))

        greedy_transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            temperature=0.0,
            extra_body=dict(seed=42))

        assert greedy_transcription.text != transcription.text


@pytest.mark.asyncio
async def test_audio_prompt(mary_had_lamb):
    model_name = "openai/whisper-large-v3-turbo"
    server_args = ["--enforce-eager"]
    prompt = "This is a speech, recorded in a phonograph."
    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        #Prompts should not omit the part of original prompt while transcribing.
        prefix = "The first words I spoke in the original phonograph"
        client = remote_server.get_async_client()
        transcription = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            temperature=0.0)
        out = json.loads(transcription)['text']
        assert prefix in out
        transcription_wprompt = await client.audio.transcriptions.create(
            model=model_name,
            file=mary_had_lamb,
            language="en",
            response_format="text",
            prompt=prompt,
            temperature=0.0)
        out_prompt = json.loads(transcription_wprompt)['text']
        assert prefix in out_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["mistralai/Voxtral-Mini-3B-2507"])
async def test_beam_search_timing(mary_had_lamb, model_name):
    """Test timing comparison between beam search and regular transcription."""
    server_args = ["--enforce-eager"]

    if model_name.startswith("mistralai"):
        server_args += MISTRAL_FORMAT_ARGS

    with RemoteOpenAIServer(model_name, server_args) as remote_server:
        client = remote_server.get_async_client()

        # Time regular transcription (5 runs)
        regular_times = []
        for i in range(5):
            mary_had_lamb.seek(0)  # Reset file pointer
            start_time = time.time()
            transcription_regular = await client.audio.transcriptions.create(
                model=model_name,
                file=mary_had_lamb,
                language="en",
                response_format="text",
                temperature=0.0,
                extra_body={
                    "use_beam_search": False,
                    "max_tokens": 300
                })
            end_time = time.time()
            regular_times.append(end_time - start_time)
            print(f"Regular transcription run {i+1}: {regular_times[-1]:.2f}s")

        # Time beam search transcription (5 runs)
        beam_times = []
        for i in range(5):
            mary_had_lamb.seek(0)  # Reset file pointer
            start_time = time.time()
            transcription_beam = await client.audio.transcriptions.create(
                model=model_name,
                file=mary_had_lamb,
                language="en",
                response_format="text",
                temperature=0.0,
                extra_body={
                    "use_beam_search": True,
                    "beam_size": 5,
                    "max_tokens": 300
                })
            end_time = time.time()
            beam_times.append(end_time - start_time)
            print(f"Beam search run {i+1}: {beam_times[-1]:.2f}s")

        # Calculate averages
        avg_regular = sum(regular_times) / len(regular_times)
        avg_beam = sum(beam_times) / len(beam_times)

        print("\nTiming Results:")
        print(f"Regular transcription average: {avg_regular:.2f}s")
        print(f"Beam search average: {avg_beam:.2f}s")
        print(f"Beam search is {avg_beam/avg_regular:.1f}x slower")
        out_regular = json.loads(transcription_regular)['text']
        out_beam = json.loads(transcription_beam)['text']
        print(f"Regular transcription: {out_regular}")
        print(f"Beam search transcription: {out_beam}")

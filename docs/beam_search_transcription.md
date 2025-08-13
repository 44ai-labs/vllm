# Beam Search for Speech-to-Text in vLLM

This feature is implemented with no changes to the core of vllm. This leads to a suboptimal performance. I call it the "budget beam search". It uses the prefix caching mechanism to maintain some amount of efficiency, but it is not as efficient as a fully optimized beam search implementation. The main goal is to provide a working solution without compromising the core architecture of vLLM.

## Overview

Beam search is now supported for speech-to-text transcription and translation endpoints. This feature allows you to generate multiple candidate transcriptions and select the best one based on cumulative log-probability scores.

## API Parameters

The following new parameters have been added to the transcription and translation endpoints:

- `use_beam_search` (bool, optional): Whether to enable beam search. Default: `False`
- `beam_size` (int, optional): Number of beams to use in beam search. Also serves as the beam size. Default: `5`

## Usage Examples

### Python with OpenAI Client

```python
import openai

client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"  # Your vLLM server
)

# Basic beam search transcription
with open("audio.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=audio_file,
        language="en",
        response_format="text",
        extra_body={
            "use_beam_search": True,
            "beam_size": 3  # Use 3 beams
        }
    )

print(transcription.text)
```

## Limitations

1. **Streaming Not Supported**: Beam search cannot be used with streaming (`stream=True`). If both are enabled, the request will be rejected with an error.

2. **Performance**: Beam search requires more computational resources as it generates multiple candidate sequences. The time complexity increases roughly linearly with the beam size.

3. **Only V1**: So whisper models are currently not supported for beam search.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the `top_logprobs` request validation on the speech-to-text
transcription and translation endpoints.

`top_logprobs` only makes sense alongside `include_token_logprobs=True`, and
must stay within the bounds enforced on the field. These are pure pydantic
validation checks, so they run without a model or server.
"""

import io

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
)
from vllm.entrypoints.speech_to_text.translation.protocol import TranslationRequest


def _upload_file() -> UploadFile:
    return UploadFile(filename="audio.wav", file=io.BytesIO(b"x"))


@pytest.mark.parametrize("request_cls", [TranscriptionRequest, TranslationRequest])
def test_top_logprobs_requires_include_token_logprobs(request_cls):
    # Raised from a `model_validator(mode="before")`, so pydantic wraps the
    # underlying VLLMValidationError in its own ValidationError.
    with pytest.raises(ValidationError, match="include_token_logprobs"):
        request_cls(file=_upload_file(), top_logprobs=5)


@pytest.mark.parametrize("request_cls", [TranscriptionRequest, TranslationRequest])
def test_top_logprobs_allowed_with_include_token_logprobs(request_cls):
    request = request_cls(
        file=_upload_file(), include_token_logprobs=True, top_logprobs=5
    )
    assert request.top_logprobs == 5


@pytest.mark.parametrize("request_cls", [TranscriptionRequest, TranslationRequest])
def test_top_logprobs_out_of_bounds_rejected(request_cls):
    with pytest.raises(ValidationError):
        request_cls(file=_upload_file(), include_token_logprobs=True, top_logprobs=21)

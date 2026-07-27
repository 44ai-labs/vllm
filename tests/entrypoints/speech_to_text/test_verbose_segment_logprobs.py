# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `SpeechToTextBaseServing._get_verbose_segments`, focused on
the optional per-token logprob and top-`k` alternative logprob fields.

`_get_verbose_segments` only touches `self.tokenizer` and its arguments, so it
is exercised directly on a bare instance (via `object.__new__`) with a fake
tokenizer, instead of standing up a full serving object (which needs a live
engine client and model registry).
"""

from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionSegment,
)
from vllm.logprobs import Logprob

# Vocab: 100/101/102 are timestamp tokens at 0.00s/0.02s/0.04s; 1/2/3/4 decode
# to "a"/"b"/"c"/"d"; 999 is eos.
_DECODE_MAP = {1: "a", 2: "b", 3: "c", 4: "d"}


class _FakeTokenizer:
    eos_token_id = 999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == "<|0.00|>"
        return [100]

    def decode(self, token_ids) -> str:
        return "".join(_DECODE_MAP.get(tid, "?") for tid in token_ids)


# Sequence: <|0.00|> a b <|0.02|><|0.02|> c <|0.04|> <eos>
# -> one degenerate empty segment (pre-existing behavior for the leading
# timestamp) followed by two real segments: "ab" and "c".
_TOKENS = (100, 1, 2, 101, 101, 3, 102, 999)
_LOG_PROBS = [
    {100: Logprob(logprob=-0.1, decoded_token="<|0.00|>")},
    {
        1: Logprob(logprob=-0.2, decoded_token="a"),
        2: Logprob(logprob=-1.5, decoded_token="b"),
    },
    {
        2: Logprob(logprob=-0.3, decoded_token="b"),
        1: Logprob(logprob=-2.0, decoded_token="a"),
    },
    {101: Logprob(logprob=-0.05, decoded_token="<|0.02|>")},
    {101: Logprob(logprob=-0.05, decoded_token="<|0.02|>")},
    {
        3: Logprob(logprob=-0.4, decoded_token="c"),
        4: Logprob(logprob=-3.0, decoded_token="d"),
    },
    {102: Logprob(logprob=-0.02, decoded_token="<|0.04|>")},
]


def _make_serving() -> SpeechToTextBaseServing:
    obj = object.__new__(SpeechToTextBaseServing)
    obj.tokenizer = _FakeTokenizer()
    return obj


def _get_segments(request: TranscriptionRequest) -> list[TranscriptionSegment]:
    return _make_serving()._get_verbose_segments(
        tokens=_TOKENS,
        log_probs=_LOG_PROBS,
        request=request,
        segment_class=TranscriptionSegment,
    )


def _upload_file():
    import io

    from fastapi import UploadFile

    return UploadFile(filename="audio.wav", file=io.BytesIO(b"x"))


def test_logprob_fields_omitted_by_default():
    request = TranscriptionRequest(file=_upload_file())
    segments = _get_segments(request)

    real_segments = [s for s in segments if s.tokens]
    assert [s.text for s in real_segments] == ["ab", "c"]
    assert all(s.token_logprobs is None for s in real_segments)
    assert all(s.top_logprobs is None for s in real_segments)


def test_token_logprobs_aligned_with_tokens():
    request = TranscriptionRequest(file=_upload_file(), include_token_logprobs=True)
    segments = _get_segments(request)

    real_segments = [s for s in segments if s.tokens]
    seg_ab, seg_c = real_segments
    assert seg_ab.token_logprobs == [-0.2, -0.3]
    assert seg_c.token_logprobs == [-0.4]
    assert all(s.top_logprobs is None for s in real_segments)


def test_top_logprobs_include_alternatives_keyed_by_decoded_token():
    request = TranscriptionRequest(
        file=_upload_file(), include_token_logprobs=True, top_logprobs=2
    )
    segments = _get_segments(request)

    real_segments = [s for s in segments if s.tokens]
    seg_ab, seg_c = real_segments

    # One dict of alternatives per token, same length as `tokens`.
    assert len(seg_ab.top_logprobs) == len(seg_ab.tokens) == 2
    assert seg_ab.top_logprobs == [{"a": -0.2, "b": -1.5}, {"b": -0.3, "a": -2.0}]

    assert len(seg_c.top_logprobs) == len(seg_c.tokens) == 1
    assert seg_c.top_logprobs == [{"c": -0.4, "d": -3.0}]

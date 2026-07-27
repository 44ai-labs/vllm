# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the flat (non-segmented) token logprob extraction used by
`response_format="text"/"json"` on transcription/translation.

This path (as opposed to `_get_verbose_segments`, which only runs for
`response_format="verbose_json"`) is what serves logprobs for models
generating without timestamps: Whisper itself always builds its prompt with
`<|notimestamps|>` unless the request is verbose_json (see
`WhisperForConditionalGeneration.get_generation_prompt`), and models like
CohereASR don't support segment timestamps (`verbose_json`) at all.
`SpeechToTextBaseServing.__init__` loads `self.tokenizer` unconditionally for
every ASR model, so both cases share the same tokenizer-backed code path.
"""

from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.logprobs import Logprob


class _FakeTokenizer:
    def decode(self, token_ids):
        return f"<decoded:{token_ids[0]}>"


def _make_serving() -> SpeechToTextBaseServing:
    obj = object.__new__(SpeechToTextBaseServing)
    obj.tokenizer = _FakeTokenizer()
    return obj


def test_returns_none_when_nothing_requested():
    obj = _make_serving()
    token_logprobs, top_logprobs = obj._extract_flat_token_logprobs(
        [1, 2],
        [{1: Logprob(logprob=-0.1)}, {2: Logprob(logprob=-0.2)}],
        include_token_logprobs=False,
        num_top_logprobs=0,
    )
    assert token_logprobs is None
    assert top_logprobs is None


def test_token_logprobs_only():
    obj = _make_serving()
    log_probs = [
        {1: Logprob(logprob=-0.1, decoded_token="a")},
        {2: Logprob(logprob=-0.2, decoded_token="b")},
    ]
    token_logprobs, top_logprobs = obj._extract_flat_token_logprobs(
        [1, 2],
        log_probs,
        include_token_logprobs=True,
        num_top_logprobs=0,
    )
    assert token_logprobs == [-0.1, -0.2]
    assert top_logprobs is None


def test_top_logprobs_use_engine_decoded_token():
    obj = _make_serving()
    log_probs = [
        {
            1: Logprob(logprob=-0.1, decoded_token="a"),
            3: Logprob(logprob=-2.0, decoded_token="c"),
        },
        {2: Logprob(logprob=-0.2, decoded_token="b")},
    ]
    token_logprobs, top_logprobs = obj._extract_flat_token_logprobs(
        [1, 2],
        log_probs,
        include_token_logprobs=True,
        num_top_logprobs=2,
    )
    assert token_logprobs == [-0.1, -0.2]
    assert top_logprobs == [{"a": -0.1, "c": -2.0}, {"b": -0.2}]


def test_top_logprobs_falls_back_to_tokenizer_decode_when_missing():
    # Defensive fallback only; the engine normally always attaches
    # `decoded_token` when a tokenizer is loaded.
    obj = _make_serving()
    log_probs = [{9: Logprob(logprob=-0.3, decoded_token=None)}]
    _, top_logprobs = obj._extract_flat_token_logprobs(
        [9],
        log_probs,
        include_token_logprobs=False,
        num_top_logprobs=1,
    )
    assert top_logprobs == [{"<decoded:9>": -0.3}]


def test_missing_position_entry_yields_nan_logprob():
    obj = _make_serving()
    token_logprobs, _ = obj._extract_flat_token_logprobs(
        [5],
        [{}],
        include_token_logprobs=True,
        num_top_logprobs=0,
    )
    assert len(token_logprobs) == 1
    assert token_logprobs[0] != token_logprobs[0]  # NaN

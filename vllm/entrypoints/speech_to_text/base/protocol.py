# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Literal, TypeAlias

import torch

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel

## Protocols for Audio
AudioResponseFormat: TypeAlias = Literal["json", "text", "srt", "verbose_json", "vtt"]
_LONG_INFO = torch.iinfo(torch.long)


class TopLogprob(OpenAIBaseModel):
    """One candidate token at one generated position.

    Identified by `id`, not by `token`: several ids can decode to the same
    string, so a text-keyed mapping silently drops candidates. SentencePiece
    tokenizers make that common rather than exotic, because decoding a single
    token strips the word-start marker — `▁de` and `de` both render as "de" and
    one overwrites the other. The evicted candidate is frequently the sampled
    token itself, which removes its probability from the response entirely and
    leaves an unrelated, much lower value in its place.
    """

    id: int
    """Token id. Unique within a position; the safe key."""

    token: str
    """Decoded text of this token. Not unique — see the class docstring."""

    logprob: float
    """Log probability of this token at this position."""

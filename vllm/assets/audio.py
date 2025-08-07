# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

import numpy.typing as npt

from vllm.utils import PlaceholderModule

from .base import (VLLM_S3_BUCKET_URL, get_44ai_public_assets,
                   get_vllm_public_assets)

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

ASSET_DIR = "multimodal_asset"

AudioAssetName = Literal["winning_call", "mary_had_lamb"]
Audio44aiName = Literal["30s_test_swiss_german_7kz53lbjqr.mp3"]


@dataclass(frozen=True)
class AudioAsset:
    name: AudioAssetName

    @property
    def filename(self) -> str:
        return f"{self.name}.ogg"

    @property
    def audio_and_sample_rate(self) -> tuple[npt.NDArray, float]:
        audio_path = get_vllm_public_assets(filename=self.filename,
                                            s3_prefix=ASSET_DIR)
        return librosa.load(audio_path, sr=None)

    def get_local_path(self) -> Path:
        return get_vllm_public_assets(filename=self.filename,
                                      s3_prefix=ASSET_DIR)

    @property
    def url(self) -> str:
        return urljoin(VLLM_S3_BUCKET_URL, f"{ASSET_DIR}/{self.name}.ogg")


@dataclass(frozen=True)
class AudioAssets44ai:
    name: Audio44aiName

    @property
    def filename(self) -> str:
        return f"{self.name}"

    @property
    def audio_and_sample_rate(self) -> tuple[npt.NDArray, float]:
        audio_path = get_44ai_public_assets(filename=self.filename)
        return librosa.load(audio_path, sr=None)

    def get_local_path(self) -> Path:
        return get_44ai_public_assets(filename=self.filename)

    @property
    def url(self) -> str:
        return urljoin(VLLM_S3_BUCKET_URL, f"{self.name}")

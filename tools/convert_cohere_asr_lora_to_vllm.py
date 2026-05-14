# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import safetensors
import torch
from safetensors.torch import save_file


LORA_WEIGHT_SUFFIXES = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_embedding_A",
    ".lora_embedding_B",
)


MAPPING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?:^|.*\.)proj_out(\.lora_(?:A|B)\.weight)$"
        ),
        r"lm_head\1",
    ),
    (
        re.compile(
            r"(?:^|.*\.)log_softmax\.mlp\.layer0(\.lora_(?:A|B)\.weight)$"
        ),
        r"lm_head\1",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.first_sub_layer\.query_net(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.first_sub_layer.q_proj\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.first_sub_layer\.key_net(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.first_sub_layer.k_proj\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.first_sub_layer\.value_net(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.first_sub_layer.v_proj\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.first_sub_layer\.out_projection(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.first_sub_layer.out_projection\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.second_sub_layer\.query_net(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.second_sub_layer.q_proj\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.second_sub_layer\.key_net(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.second_sub_layer.k_proj\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.second_sub_layer\.value_net(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.second_sub_layer.v_proj\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.second_sub_layer\.out_projection(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.second_sub_layer.out_projection\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.third_sub_layer\.dense_in(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.third_sub_layer.dense_in\3",
    ),
    (
        re.compile(
            r"(^|.*\.)transf_decoder\._decoder\.layers\.(\d+)"
            r"\.third_sub_layer\.dense_out(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1model.decoder.layers.\2.third_sub_layer.dense_out\3",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.self_attn\.q_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.first_sub_layer.q_proj\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.self_attn\.k_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.first_sub_layer.k_proj\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.self_attn\.v_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.first_sub_layer.v_proj\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.self_attn\.o_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.first_sub_layer.out_projection\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.encoder_attn\.q_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.second_sub_layer.q_proj\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.encoder_attn\.k_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.second_sub_layer.k_proj\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.encoder_attn\.v_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.second_sub_layer.v_proj\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.encoder_attn\.o_proj"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.second_sub_layer.out_projection\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.mlp(?:\.mlp)?\.fc1"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.third_sub_layer.dense_in\2",
    ),
    (
        re.compile(
            r"(\.decoder\.layers\.\d+)\.mlp(?:\.mlp)?\.fc2"
            r"(\.lora_(?:A|B)\.weight)$"
        ),
        r"\1.third_sub_layer.dense_out\2",
    ),
)


def is_lora_tensor(name: str) -> bool:
    return name.endswith(LORA_WEIGHT_SUFFIXES)


def map_tensor_name(name: str) -> str | None:
    if not is_lora_tensor(name):
        return None

    if ".encoder." in name:
        return name

    for pattern, replacement in MAPPING_RULES:
        if pattern.search(name):
            return pattern.sub(replacement, name)

    return None


def target_suffix(tensor_name: str) -> str:
    for suffix in LORA_WEIGHT_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)].rsplit(".", 1)[-1]
    raise ValueError(f"Unsupported LoRA tensor name: {tensor_name}")


def find_weights_path(checkpoint_dir: Path) -> Path:
    for file_name in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
    ):
        candidate = checkpoint_dir / file_name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No adapter weights found under {checkpoint_dir}")


def load_tensors(weights_path: Path) -> dict[str, torch.Tensor]:
    if weights_path.suffix == ".safetensors":
        tensors: dict[str, torch.Tensor] = {}
        with safetensors.safe_open(weights_path, framework="pt") as adapter_file:  # type: ignore[arg-type]
            for key in adapter_file.keys():
                tensors[key] = adapter_file.get_tensor(key)
        return tensors
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}-vllm")


def convert_checkpoint(input_dir: Path, output_dir: Path, overwrite: bool) -> tuple[int, int]:
    weights_path = find_weights_path(input_dir)
    tensors = load_tensors(weights_path)

    converted_tensors: dict[str, torch.Tensor] = {}
    unsupported: list[str] = []

    for name, tensor in tensors.items():
        if not is_lora_tensor(name):
            continue
        mapped_name = map_tensor_name(name)
        if mapped_name is None:
            unsupported.append(name)
            continue
        converted_tensors[mapped_name] = tensor

    if not converted_tensors:
        raise ValueError(
            "No decoder-side Cohere ASR LoRA tensors were recognized in the checkpoint."
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
    else:
        output_dir.mkdir(parents=True)

    save_file(converted_tensors, output_dir / "adapter_model.safetensors")

    config_path = input_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        adapter_config: dict[str, Any] = json.load(config_file)

    adapter_config["target_modules"] = sorted(
        {target_suffix(name) for name in converted_tensors}
    )

    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(adapter_config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")

    return len(converted_tensors), len(unsupported)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Cohere ASR LoRA checkpoint into a decoder-only vLLM-compatible adapter."
        )
    )
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to the source checkpoint directory, for example checkpoint-50000.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to checkpoint-xxxx-vllm next to the input.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.checkpoint_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(input_dir)
    )

    converted_count, unsupported_count = convert_checkpoint(
        input_dir,
        output_dir,
        overwrite=args.overwrite,
    )

    print(f"input: {input_dir}")
    print(f"output: {output_dir}")
    print(f"converted tensors: {converted_count}")
    print(f"dropped unsupported tensors: {unsupported_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal client for Cohere Transcribe with a served LoRA adapter.

Run against an already running server:

    python examples/speech_to_text/openai/cohere_transcribe_lora_client.py \
        --base-url http://127.0.0.1:8001/v1

Or let this script launch `vllm serve` itself:

    python examples/speech_to_text/openai/cohere_transcribe_lora_client.py \
        --launch-server --server-port 8001
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import APIConnectionError, OpenAI


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "cohere-transcribe-lora"
DEFAULT_AUDIO_PATH = "/scratch/44ai-train/p44ai_audio/demo_kispi.wav"
DEFAULT_BASE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_ADAPTER_PATH = "/scratch/44ai-train/eval/cohere/eval_cohere_big_v3/checkpoint-50000-vllm"
DEFAULT_HF_TOKEN = "SET_LE_TOKEN"
DEFAULT_USE_FLASHINFER_SAMPLER = False
DEFAULT_TRUST_REMOTE_CODE = False


def get_vllm_bin() -> str:
    vllm_bin = Path(sys.executable).with_name("vllm")
    if vllm_bin.is_file():
        return str(vllm_bin)

    resolved = shutil.which("vllm")
    if resolved is None:
        raise FileNotFoundError("Could not locate the vllm executable.")
    return resolved


def launch_server(args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["HF_TOKEN"] = args.hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if args.use_flashinfer_sampler else "0"
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    command = [
        get_vllm_bin(),
        "serve",
        args.base_model,
        "--port",
        str(args.server_port),
        "--enable-lora",
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-log-requests",
        "--lora-modules",
        (
            "{"
            f'"name":"{args.model}",'
            f'"path":"{args.adapter_path}",'
            f'"base_model_name":"{args.base_model}"'
            "}"
        ),
    ]

    if args.trust_remote_code:
        command.append("--trust-remote-code")
    else:
        command.append("--no-trust-remote-code")

    return subprocess.Popen(command, env=env)


def wait_for_server(
    base_url: str,
    api_key: str,
    timeout_s: float = 600.0,
) -> None:
    client = OpenAI(api_key=api_key, base_url=base_url)
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        try:
            client.models.list()
            return
        except APIConnectionError:
            time.sleep(2.0)

    raise TimeoutError(f"Timed out waiting for server at {base_url}")


def get_served_model_ids(client: OpenAI) -> list[str]:
    return [model.id for model in client.models.list().data]


def transcribe_once(
    client: OpenAI,
    *,
    audio_path: Path,
    model: str,
    language: str,
    response_format: str,
    temperature: float,
    prompt: str,
    hotwords: str | None,
    seed: int | None,
):
    extra_body: dict[str, str | int] | None = None
    if hotwords:
        extra_body = {"hotwords": hotwords}
    if seed is not None:
        if extra_body is None:
            extra_body = {}
        extra_body["seed"] = seed

    with audio_path.open("rb") as audio_file:
        return client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            prompt=prompt,
            extra_body=extra_body,
        )


def print_response(label: str, response) -> None:
    print(f"\n=== {label} ===")
    if hasattr(response, "text"):
        print(response.text)
    else:
        print(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio through a vLLM-served Cohere LoRA model."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible vLLM server base URL.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--hf-token",
        default=DEFAULT_HF_TOKEN,
        help="Hugging Face token used when launching the gated base model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Served model ID. Use the LoRA adapter name when statically loaded.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model to serve when using --launch-server.",
    )
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_ADAPTER_PATH,
        help=(
            "Local LoRA adapter path to load into the server. Raw Cohere ASR "
            "PEFT adapters may need conversion with "
            "tools/convert_cohere_asr_lora_to_vllm.py first."
        ),
    )
    parser.add_argument(
        "--audio-path",
        default=DEFAULT_AUDIO_PATH,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch a local vLLM server before sending the transcription request.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port to use when launching the local vLLM server.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=256,
        help="Maximum LoRA rank to configure for the launched server.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization to use when launching the local vLLM server.",
    )
    parser.add_argument(
        "--use-flashinfer-sampler",
        action="store_true",
        default=DEFAULT_USE_FLASHINFER_SAMPLER,
        help="Enable the FlashInfer sampler when launching the local vLLM server.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRUST_REMOTE_CODE,
        help="Whether to pass --trust-remote-code when launching vLLM serve.",
    )
    parser.add_argument(
        "--language",
        default="de",
        help="ISO-639-1 language code for the transcription request.",
    )
    parser.add_argument(
        "--response-format",
        default="json",
        choices=["json", "text", "srt", "verbose_json", "vtt"],
        help="Output format for the transcription API.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help="Optional hotwords string passed through the transcription API.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional prompt passed through the transcription API.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional seed sent to both transcription requests for repeatability.",
    )
    parser.add_argument(
        "--compare-base-and-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run one transcription with the LoRA-served model and one with the "
            "base model against the same running server."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    server_process: subprocess.Popen[str] | None = None
    if args.launch_server:
        if args.base_url == DEFAULT_BASE_URL:
            args.base_url = f"http://127.0.0.1:{args.server_port}/v1"
        server_process = launch_server(args)
        wait_for_server(args.base_url, args.api_key)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    try:
        served_model_ids = get_served_model_ids(client)
        print("Served models:")
        for model_id in served_model_ids:
            print(f"- {model_id}")

        lora_active = args.model in served_model_ids
        print(f"\nLoRA model visible to server: {lora_active}")

        lora_response = transcribe_once(
            client,
            audio_path=audio_path,
            model=args.model,
            language=args.language,
            response_format=args.response_format,
            temperature=args.temperature,
            prompt=args.prompt,
            hotwords=args.hotwords,
            seed=args.seed,
        )
        print_response(f"Transcription via {args.model}", lora_response)

        if args.compare_base_and_lora:
            base_response = transcribe_once(
                client,
                audio_path=audio_path,
                model=args.base_model,
                language=args.language,
                response_format=args.response_format,
                temperature=args.temperature,
                prompt=args.prompt,
                hotwords=args.hotwords,
                seed=args.seed,
            )
            print_response(
                f"Transcription via {args.base_model}",
                base_response,
            )
    finally:
        if server_process is not None:
            server_process.terminate()
            server_process.wait(timeout=30)


if __name__ == "__main__":
    main()
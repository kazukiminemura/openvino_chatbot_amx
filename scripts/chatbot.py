#!/usr/bin/env python
"""Interactive console chatbot demo powered by OpenVINO + Intel AMX."""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from threading import Thread
from typing import Dict, Iterable, List

from rich.console import Console
from rich.markdown import Markdown
from transformers import AutoTokenizer, TextIteratorStreamer

from optimum.intel.openvino import OVModelForCausalLM

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenVINO-based INT8 chatbot loop.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/int8_chatbot_model"),
        help="Directory containing the exported OpenVINO model (model.xml / openvino_model.xml).",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        help="OpenVINO device to target (CPU enables Intel AMX on supported Xeon / Core Ultra).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per assistant turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling cut-off.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are an efficient assistant optimized for Intel AMX and OpenVINO. "
            "Answer concisely and stay helpful."
        ),
        help="System prompt injected at the beginning of the dialogue.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of previous turns to include when building the prompt context.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming and print responses once generation completes.",
    )
    parser.add_argument(
        "--compile-config",
        type=str,
        default=None,
        help="Optional path to an ov_config.json file overriding compiled runtime hints.",
    )
    return parser.parse_args()


def load_ov_runtime_config(model_dir: Path, override_path: str | None) -> Dict[str, str]:
    candidates: List[Path] = []
    if override_path is not None:
        candidates.append(Path(override_path))
    candidates.append(model_dir / "ov_config.json")

    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            return {str(k): str(v) for k, v in config.items()}

    return {
        "INFERENCE_PRECISION_HINT": "int8",
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
    }


def build_prompt(system_prompt: str, dialogue: List[Dict[str, str]], tokenizer) -> str:
    history = []
    history.append(f"<|system|>\n{system_prompt}\n<|end|>")
    for turn in dialogue:
        role = turn["role"]
        content = turn["content"]
        history.append(f"<|{role}|>\n{content}\n<|end|>")
    history.append("<|assistant|>\n")
    prompt = "".join(history)

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            *dialogue,
        ], tokenize=False, add_generation_prompt=True)
    return prompt


def stream_generate(model, tokenizer, prompt: str, args: argparse.Namespace) -> Iterable[str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    worker = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    worker.start()
    for new_text in streamer:
        yield new_text
    worker.join()


def full_generate(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    generation = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated_ids = generation[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        console.print(f"[red]Model directory {model_dir} not found. Run scripts/prepare_model.py first.[/red]")
        sys.exit(1)

    runtime_config = load_ov_runtime_config(model_dir, args.compile_config)
    console.print("[bold green]Loading tokenizer...[/bold green]")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token or tokenizer.unk_token or "</s>"

    console.print("[bold green]Loading OpenVINO model (INT8)...[/bold green]")
    model = OVModelForCausalLM.from_pretrained(
        model_dir,
        compile=True,
        device=args.device,
        ov_config=runtime_config,
        trust_remote_code=True,
    )

    console.print("[cyan]Ready. Type 'exit', 'quit', or press Ctrl+C to stop.[/cyan]")
    dialogue: List[Dict[str, str]] = []

    def signal_handler(signum, frame):  # noqa: ARG001
        console.print("\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        console.print("\n[bold blue]User:[/bold blue] ", end="", highlight=False)
        try:
            user_input = input()
        except EOFError:
            break

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            break
        if not user_input.strip():
            continue

        dialogue.append({"role": "user", "content": user_input.strip()})
        if len(dialogue) > args.history * 2:
            dialogue = dialogue[-args.history * 2 :]

        prompt = build_prompt(args.system_prompt, dialogue, tokenizer)
        console.print("[bold magenta]Assistant:[/bold magenta] ", end="", highlight=False)

        if args.no_stream:
            answer = full_generate(model, tokenizer, prompt, args)
            console.print(Markdown(answer.strip() or "(empty response)"))
        else:
            buffer = []
            try:
                for token in stream_generate(model, tokenizer, prompt, args):
                    buffer.append(token)
                    console.print(token, end="", highlight=False, soft_wrap=True)
                console.print()
            except KeyboardInterrupt:
                console.print("[yellow]\nGeneration interrupted.[/yellow]")
            answer = "".join(buffer).strip()
        dialogue.append({"role": "assistant", "content": answer})

    console.print("[bold green]\nSession finished.[/bold green]")


if __name__ == "__main__":
    main()

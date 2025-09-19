#!/usr/bin/env python
"""Interactive console chatbot demo powered by OpenVINO + Intel AMX."""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Dict, Iterable, List, Optional

import numpy as np
from rich.console import Console
from rich.markdown import Markdown
from transformers import AutoTokenizer, TextIteratorStreamer
try:
    from transformers.generation.logits_process import top_k_top_p_filtering
except ImportError:
    try:
        from transformers.generation.utils import top_k_top_p_filtering
    except ImportError:
        top_k_top_p_filtering = None

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional until native backend is requested
    torch = None

if top_k_top_p_filtering is None:
    def _fallback_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
        if torch is None:
            raise RuntimeError("PyTorch is required for the native OpenVINO backend. Install torch to proceed.")
        logits = logits.clone()
        original_dtype = logits.dtype
        logits = logits.to(torch.float32)
        top_k = int(top_k) if top_k else 0
        if top_k > 0 and top_k < logits.size(-1):
            values, _ = torch.topk(logits, top_k)
            kth = values[..., -1, None]
            mask = logits < kth
            logits = logits.masked_fill(mask, filter_value)
        top_p = float(top_p)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, filter_value)
            logits = torch.full_like(logits, filter_value)
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return logits.to(original_dtype)
    top_k_top_p_filtering = _fallback_top_k_top_p_filtering

console = Console()

_OV_TO_NUMPY_DTYPE = {
    "f32": np.float32,
    "f16": np.float16,
    "bf16": np.float32,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "u64": np.uint64,
    "u32": np.uint32,
    "u16": np.uint16,
    "u8": np.uint8,
    "boolean": np.bool_,
}


def count_generated_tokens(text: str, tokenizer) -> int:
    """Return the number of tokens in `text` according to the tokenizer."""
    if not text:
        return 0
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except Exception:  # pragma: no cover - just defensive
        return 0
    return len(tokens)


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
    parser.add_argument(
        "--backend",
        choices=("optimum", "native"),
        default="optimum",
        help="Generation backend: 'optimum' uses Optimum-Intel, 'native' runs directly on openvino.runtime.",
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


def optimum_stream_generate(model, tokenizer, prompt: str, args: argparse.Namespace) -> Iterable[str]:
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


def optimum_full_generate(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
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


def resolve_model_xml(model_dir: Path) -> Path:
    candidates = [
        model_dir / "openvino_model.xml",
        model_dir / "model.xml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    xml_files = sorted(model_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No OpenVINO IR (.xml) found in {model_dir}.")
    if len(xml_files) > 1:
        raise RuntimeError(
            "Multiple .xml files detected. Please keep a single OpenVINO IR in --model-dir or specify a subdirectory."
        )
    return xml_files[0]


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    if attention_mask is None:
        raise ValueError("attention_mask is required to derive position_ids")
    cumulative = np.cumsum(attention_mask, axis=-1)
    position_ids = cumulative - 1
    position_ids = np.clip(position_ids, a_min=0, a_max=None)
    position_ids = np.where(attention_mask == 0, 0, position_ids)
    return position_ids.astype(np.int64)


def _sample_next_token(logit_slice: np.ndarray, temperature: float, top_p: float) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for the native OpenVINO backend. Install torch to proceed.")
    logits = torch.from_numpy(logit_slice).to(torch.float32)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if temperature <= 1e-5:
        return int(torch.argmax(logits, dim=-1)[0].item())

    logits = logits / max(temperature, 1e-5)
    filtered = top_k_top_p_filtering(logits, top_k=0, top_p=max(min(top_p, 1.0), 0.0))
    probs = torch.nn.functional.softmax(filtered, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token[0, 0].item())


class NativeOVPipeline:
    def __init__(self, model_dir: Path, device: str, runtime_config: Dict[str, str]):
        try:
            import openvino as ov
        except ImportError as exc:  # pragma: no cover - executed only when openvino is missing
            raise RuntimeError(
                "The 'openvino' package is required for --backend native. Install it with 'pip install openvino'."
            ) from exc

        xml_path = resolve_model_xml(model_dir)
        self.core = ov.Core()
        self.model = self.core.read_model(xml_path)
        self.compiled = self.core.compile_model(self.model, device, runtime_config)
        self.request = self.compiled.create_infer_request()

        self.input_ports = {port.get_any_name(): port for port in self.compiled.inputs}
        self.output_ports = {port.get_any_name(): port for port in self.compiled.outputs}
        self.input_dtypes = {name: self._dtype_for_port(port) for name, port in self.input_ports.items()}
        self.output_dtypes = {name: self._dtype_for_port(port) for name, port in self.output_ports.items()}

        self.input_ids_name = self._find_input("input_ids")
        self.attention_mask_name = self._find_input("attention_mask", required=False)
        self.position_ids_name = self._find_input("position_ids", required=False)
        self.beam_idx_name = self._find_input("beam_idx", required=False)

        self.requires_attention_mask = self.attention_mask_name is not None
        self.requires_position_ids = self.position_ids_name is not None
        self.requires_beam_idx = self.beam_idx_name is not None

        self.cache_input_names = sorted(
            [name for name in self.input_ports if "past_key_values" in name],
            key=self._cache_key,
        )
        self.cache_output_names = sorted(
            [name for name in self.output_ports if "present" in name or "past_key_values" in name],
            key=self._cache_key,
        )

        self.cache_pairs = self._build_cache_pairs()
        self.logits_name = self._find_logits_output()
        self.reset()

    def _find_input(self, needle: str, required: bool = True) -> Optional[str]:
        for name in self.input_ports:
            if needle in name:
                return name
        if required:
            raise RuntimeError(f"Model input containing '{needle}' was not found in the OpenVINO IR.")
        return None

    def _find_logits_output(self) -> str:
        for name in self.output_ports:
            if "logits" in name:
                return name
        raise RuntimeError("Model output containing 'logits' was not found.")

    @staticmethod
    def _cache_key(name: str) -> tuple[int, int, str]:
        match = re.search(r"(\d+)", name)
        layer = int(match.group()) if match else 0
        lower = name.lower()
        if "key" in lower:
            suffix = 0
        elif "value" in lower:
            suffix = 1
        else:
            suffix = 2
        return (layer, suffix, lower)

    def _build_cache_pairs(self) -> List[tuple[str, str]]:
        if not self.cache_input_names or not self.cache_output_names:
            return []
        input_map = {self._cache_key(name): name for name in self.cache_input_names}
        output_map = {self._cache_key(name): name for name in self.cache_output_names}
        pairs = []
        for key, input_name in input_map.items():
            output_name = output_map.get(key)
            if output_name:
                pairs.append((input_name, output_name))
        return pairs

    def _dtype_for_port(self, port) -> np.dtype:
        type_name = port.get_element_type().get_type_name()
        if type_name not in _OV_TO_NUMPY_DTYPE:
            raise RuntimeError(
                f"Unsupported OpenVINO element type '{type_name}' for input {port.get_any_name()}."
            )
        return _OV_TO_NUMPY_DTYPE[type_name]

    def _shape_for_port(self, port) -> tuple[int, ...]:
        shape = []
        for dim in port.get_partial_shape():
            if dim.is_dynamic:
                shape.append(0)
            else:
                shape.append(int(dim.get_length()))
        return tuple(shape)

    def _ensure_dtype(self, name: str, value: np.ndarray) -> np.ndarray:
        expected = self.input_dtypes.get(name)
        if expected is None:
            return value
        if value.dtype != expected:
            return value.astype(expected, copy=False)
        return value

    def reset(self) -> None:
        self.cache: Dict[str, np.ndarray] = {}
        for name in self.cache_input_names:
            port = self.input_ports[name]
            dtype = self._dtype_for_port(port)
            shape = self._shape_for_port(port)
            self.cache[name] = np.zeros(shape, dtype=dtype)

    def infer(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray],
        position_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        feeds: Dict[str, np.ndarray] = {
            self.input_ids_name: self._ensure_dtype(self.input_ids_name, input_ids),
        }
        if self.requires_attention_mask and attention_mask is not None:
            feeds[self.attention_mask_name] = self._ensure_dtype(self.attention_mask_name, attention_mask)
        if self.requires_position_ids and position_ids is not None:
            feeds[self.position_ids_name] = self._ensure_dtype(self.position_ids_name, position_ids)
        if self.requires_beam_idx:
            batch = int(input_ids.shape[0])
            beam_idx = np.arange(batch, dtype=self.input_dtypes[self.beam_idx_name])
            feeds[self.beam_idx_name] = self._ensure_dtype(self.beam_idx_name, beam_idx)

        feeds.update(self.cache)
        self.request.infer(feeds)
        outputs: Dict[str, np.ndarray] = {}
        for port in self.compiled.outputs:
            name = port.get_any_name()
            ov_tensor = self.request.get_tensor(port)
            array = np.asarray(getattr(ov_tensor, 'data', ov_tensor), dtype=self.output_dtypes[name])
            outputs[name] = array
        self._update_cache(outputs)
        return outputs[self.logits_name]

    def _update_cache(self, outputs: Dict[str, np.ndarray]) -> None:
        if not self.cache_pairs:
            return
        for input_name, output_name in self.cache_pairs:
            if output_name in outputs:
                self.cache[input_name] = self._ensure_dtype(input_name, outputs[output_name])


def native_stream_generate(
    pipeline: NativeOVPipeline,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
) -> Iterable[str]:
    if torch is None:
        raise RuntimeError("PyTorch is required for the native OpenVINO backend. Install torch to proceed.")

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cpu").numpy()

    attention_mask_tensor = encoded.get("attention_mask")
    if pipeline.requires_attention_mask:
        if attention_mask_tensor is not None:
            attention_mask = attention_mask_tensor.to("cpu").numpy()
        else:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
    else:
        attention_mask = None

    if pipeline.requires_position_ids:
        position_ids = _compute_position_ids(attention_mask)
    else:
        position_ids = None

    pipeline.reset()
    logits = pipeline.infer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    generated_ids: List[int] = []
    decoded_so_far = ""
    eos_token_id = tokenizer.eos_token_id
    total_length = input_ids.shape[1]

    for _ in range(args.max_new_tokens):
        if logits.ndim == 3:
            next_token_logits = logits[:, -1, :]
        else:
            next_token_logits = logits
        next_token_id = _sample_next_token(next_token_logits, args.temperature, args.top_p)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        generated_ids.append(int(next_token_id))
        total_length += 1
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_piece = decoded_text[len(decoded_so_far):]
        decoded_so_far = decoded_text
        yield new_piece

        input_ids = np.array([[next_token_id]], dtype=np.int64)
        if pipeline.requires_attention_mask and attention_mask is not None:
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=attention_mask.dtype)],
                axis=1,
            )
        if pipeline.requires_position_ids:
            position_ids = np.array([[total_length - 1]], dtype=np.int64)

        logits = pipeline.infer(
            input_ids=input_ids,
            attention_mask=attention_mask if pipeline.requires_attention_mask else None,
            position_ids=position_ids if pipeline.requires_position_ids else None,
        )


def native_full_generate(
    pipeline: NativeOVPipeline,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    return "".join(native_stream_generate(pipeline, tokenizer, prompt, args)).strip()


def main() -> None:
    args = parse_args()

    if args.backend == "native" and torch is None:
        console.print(
            "[red]PyTorch is required for the native backend. Install torch before retrying.[/red]"
        )
        sys.exit(1)

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        console.print(f"[red]Model directory {model_dir} not found. Run scripts/prepare_model.py first.[/red]")
        sys.exit(1)

    runtime_config = load_ov_runtime_config(model_dir, args.compile_config)
    console.print("[bold green]Loading tokenizer...[/bold green]")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token or tokenizer.unk_token or "</s>"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    backend_choice = args.backend.lower()
    if backend_choice == "optimum":
        console.print("[bold green]Loading OpenVINO model (Optimum backend)...[/bold green]")
        try:
            from optimum.intel.openvino import OVModelForCausalLM
        except ImportError:
            console.print(
                "[red]optimum-intel is required for --backend optimum. Install it with 'pip install optimum-intel[openvino]'.[/red]"
            )
            sys.exit(1)
        model = OVModelForCausalLM.from_pretrained(
            model_dir,
            compile=True,
            device=args.device,
            ov_config=runtime_config,
            trust_remote_code=True,
        )
        stream_fn = lambda prompt: optimum_stream_generate(model, tokenizer, prompt, args)
        full_fn = lambda prompt: optimum_full_generate(model, tokenizer, prompt, args)
    else:
        console.print("[bold green]Loading OpenVINO model (native backend)...[/bold green]")
        try:
            pipeline = NativeOVPipeline(model_dir, args.device, runtime_config)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        stream_fn = lambda prompt: native_stream_generate(pipeline, tokenizer, prompt, args)
        full_fn = lambda prompt: native_full_generate(pipeline, tokenizer, prompt, args)

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
        start_time = time.perf_counter()

        if args.no_stream:
            try:
                answer = full_fn(prompt)
            except KeyboardInterrupt:
                console.print("[yellow]\nGeneration interrupted.[/yellow]")
                answer = ""
            console.print(Markdown(answer.strip() or "(empty response)"))
        else:
            buffer: List[str] = []
            try:
                for token_text in stream_fn(prompt):
                    buffer.append(token_text)
                    console.print(token_text, end="", highlight=False, soft_wrap=True)
                console.print()
            except KeyboardInterrupt:
                console.print("[yellow]\nGeneration interrupted.[/yellow]")
            answer = "".join(buffer).strip()

        elapsed = time.perf_counter() - start_time
        token_count = count_generated_tokens(answer, tokenizer)
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0.0
        console.print(f"[dim green]Tokens: {token_count} | Time: {elapsed:.2f}s | Rate: {tokens_per_second:.2f} tok/s[/dim green]")
        dialogue.append({"role": "assistant", "content": answer})

    console.print("[bold green]\nSession finished.[/bold green]")


if __name__ == "__main__":
    main()




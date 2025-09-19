# OpenVINO AMX Chatbot Demo

Intel AMX enables high-throughput INT8 matrix math on recent Intel CPUs, and OpenVINO exposes those instructions to Python with minimal code. This repository delivers an end-to-end workflow that:

- exports and quantizes a Hugging Face instruction-tuned language model to OpenVINO IR (INT8)
- configures OpenVINO runtime hints so CPU inference can take advantage of AMX
- runs a terminal-based chatbot loop powered by the exported model

> Note: The workflow targets Windows or Linux hosts with Intel(R) AMX capable processors (Intel Xeon Sapphire Rapids or newer, and Intel Core Ultra). Systems without AMX still run the demo but fall back to other CPU kernels.

## Repository layout

```
requirements.txt          # Python dependencies (OpenVINO + optimum-intel + tooling)
scripts/prepare_model.py  # Wraps optimum-cli export -> OpenVINO INT8 artifacts
scripts/chatbot.py        # Interactive REPL that streams responses from the INT8 model
```

## 1. Environment setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (recommended):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   ```
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the required packages:
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

## 2. Enable AMX-friendly runtime hints (optional but recommended)

Set the following environment variables before launching any Python script. They hint OpenVINO to pick INT8 kernels that map onto AMX tiles.

```powershell
set OV_CPU_CAPABILITY_POLICY=AMX
set OV_CPU_HINT_ENABLE_AMX_INT8=1
```
```bash
export OV_CPU_CAPABILITY_POLICY=AMX
export OV_CPU_HINT_ENABLE_AMX_INT8=1
```

OpenVINO chooses the best instruction set automatically, so these settings are not strictly required; they simply make the intent explicit.

## 3. Export or quantize a model to OpenVINO INT8

`scripts/prepare_model.py` wraps `optimum-cli` so you do not have to memorize its arguments.

```bash
python scripts/prepare_model.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir artifacts/int8_chatbot_model \
  --precision int8
```

Key flags:

- `--model-id` - any causal LM hosted on the Hugging Face Hub. Swap in larger models if you have the RAM.
- `--precision` - `int8` leverages AMX; `int4` uses hybrid int4 and int8 weights; `bf16` keeps higher precision while still benefiting from AMX bfloat16 kernels.
- `--task` - override when exporting GPT2-like models (`text-generation`).
- `--trust-remote-code` - forward if the model repo requires custom modeling code.

The script drops runtime hints into `ov_config.json`, which the chatbot uses when compiling the OpenVINO model.

## 4. Run the AMX-accelerated chatbot

After the export step finishes, launch the interactive REPL:

```bash
python scripts/chatbot.py --model-dir artifacts/int8_chatbot_model
```

Useful options:

- `--temperature` and `--top-p` - control sampling.
- `--max-new-tokens` - clamp generation length.
- `--no-stream` - print the answer only when the generation call finishes.
- `--system-prompt` - inject your own system message.

Type `exit` or press `Ctrl+C` to end the session. On first run, OpenVINO compiles the network; subsequent runs reuse the cached blobs stored inside `ov_cache/`.

## 5. Troubleshooting

- **optimum-cli not found** - install `optimum-intel`: `pip install optimum-intel[openvino]`.
- **Model requires custom code** - re-run `prepare_model.py` with `--trust-remote-code`.
- **Slow performance** - ensure the environment variables from step 2 are set and NUMA pinning is enabled. You can also tune threads via `--compile-config` and `NUM_STREAMS` in `ov_config.json`.
- **Insufficient RAM** - choose a smaller model (for example `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).

## 6. Next steps

- Add a Gradio or FastAPI frontend on top of `scripts/chatbot.py` for a web demo.
- Extend `prepare_model.py` to run calibration-aware quantization (`nncf`) on a custom dataset for better accuracy.
- Integrate telemetry (Intel VTune, OpenVINO profiling) to measure AMX utilization in detail.

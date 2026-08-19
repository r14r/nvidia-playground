# NVIDIA NIM Playground

Version **0.7.2**

## Navigation

```text
Run
  ├── Single model
  └── Multiple models

Models
  ├── Info
  ├── Metadata
  └── Catalog
```

## Run modes

### Single model
Runs one prompt against one selected NVIDIA NIM model. Results are shown in **Response** and **Inspector** tabs.

### Multiple models
Runs the same prompt against all selected models concurrently. Each model is executed as its own native `asyncio.Task` using `AsyncOpenAI`.

The first result tab is **Status**, followed by one tab per model. Each model tab contains **Response** and **Inspector**.

## Multi-model live rendering

Version 0.7.2 moves the asyncio event loop into a dedicated background thread while Streamlit rendering stays on the main script thread. Every streaming content delta is forwarded immediately and updates the corresponding model's **Response** tab while the remaining model tasks continue running.

The default **Max Tokens** generation budget is now **16384**. Reasoning-capable models can consume reasoning tokens from this same budget before final answer content is emitted. The previous application default of 2048 is migrated automatically.

`nvidia-lib` version **0.4.2** also normalizes OpenAI-compatible list-based content blocks into one complete response string.

## Runtime settings

The Run sidebar contains:

- Model / Models
- Temperature
- Top P
- Max Tokens
- Streaming
- Raw chunk data

`Top P` is restricted to `0.05..1.0`; non-positive values passed directly to the library are omitted so the provider can use its default.

## Console output

Prompt runs print the selected model, System Prompt, User Prompt and final assembled response to the terminal running Streamlit. Individual streaming chunks are not printed.

## Run

```bash
just run
```

## update-cli

Project setup is configured in `update-cli.yaml`.

```bash
just setup-update-cli
just run-update-cli
```

The `run` task executes:

```bash
.venv/bin/streamlit run app/app.py
```

## Model catalog

```bash
just models
```

executes:

```bash
nvidia-cli models --list --details --with-api-key --json --save models.json
```

`models.json` and `.env` are intentionally excluded from Git.

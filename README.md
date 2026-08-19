# NVIDIA NIM Playground

Version **0.7.6**

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
Runs the same prompt against all selected models concurrently. Each selected model is executed as its own `asyncio.Task`. The first result tab is **Status**, followed by one tab per model; each model tab contains **Response** and **Inspector**.

Blocking requests reuse the synchronous `query()` path through `asyncio.to_thread()`. Streaming requests reuse `stream_events()` and forward every received event to the corresponding Response tab.

## Runtime settings

The Run sidebar contains Model / Models, Temperature, Top P, Max Tokens, Streaming and Raw chunk data. `Top P` is restricted to `0.05..1.0`; the default Max Tokens value is 2048.

## Default prompts

System Prompt:

```text
You are a helpful technical assistant.
```

User Prompt:

```text
**Describe advanced Streamlit features in three sentences per feature. No further information or additional details.**
```

## Console output

Prompt runs print model, prompts, execution lifecycle and the final assembled response to the terminal running Streamlit. Individual streaming chunks are not printed.

```text
2026-08-19T19:09:30.614+02:00 Run Prompt on Model: meta/llama-3.3-70b-instruct
2026-08-19T19:09:30.614+02:00 System Prompt: "You are a helpful technical assistant."
2026-08-19T19:09:30.614+02:00 User Prompt: "**Describe advanced Streamlit features in three sentences per feature. No further information or additional details.**"
2026-08-19T19:09:30.614+02:00  Connect to NVIDIA
2026-08-19T19:09:30.614+02:00  Run Prompt
2026-08-19T19:09:30.614+02:00  Waiting for Response
2026-08-19T19:09:30.614+02:00  Response:
```

## Run

```bash
just run
```

or directly:

```bash
streamlit run app/app.py
```

## Application helper layout

Shared Streamlit helpers live in `app/lib/`:

```text
app/lib/
├── console.py
├── run_common.py
└── shared.py
```

`app/lib/` intentionally has **no `__init__.py`**. This is required because the NVIDIA client library uses the import namespace `lib.nvidia`; making `app/lib` a regular top-level Python package named `lib` can shadow `lib/src/lib/nvidia` when Streamlit puts `app/` on `sys.path`.

`app/lib/shared.py` also adds the repository-local `lib/src` directory to `sys.path` before importing `lib.nvidia`, so direct Streamlit startup works even if the editable package has not yet been installed.

## update-cli

Project setup is configured in `update-cli.yaml`.

```bash
just setup-update-cli
just run-update-cli
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

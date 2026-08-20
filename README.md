# NVIDIA NIM Playground

Version **0.7.7**

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
Runs the same prompt against all selected models. The first result tab is **Status**, followed by one tab per model; each model tab contains **Response** and **Inspector**.

The sidebar contains **Run Parallel**:

- enabled: selected models run concurrently in separate worker threads using `ThreadPoolExecutor`
- disabled: selected models run sequentially, one after another

The Multiple Models page does **not** use `asyncio` for prompt execution. Both modes use the same synchronous NVIDIA client methods as Single Model: `query()` for blocking requests and `stream_events()` for streaming requests.

Streaming events are forwarded through a thread-safe queue to the Streamlit main thread so each Response tab can update while its model is running.

## Runtime settings

The Run sidebar contains Model / Models, Temperature, Top P, Max Tokens, Streaming and Raw chunk data. Multiple Models additionally contains **Run Parallel**. `Top P` is restricted to `0.05..1.0`; the default Max Tokens value is 2048.

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

`app/lib/` intentionally has **no `__init__.py`** because the NVIDIA client library uses the import namespace `lib.nvidia`. `app/lib/shared.py` adds the repository-local `lib/src` directory to `sys.path` before importing `lib.nvidia`.

## NVIDIA client library

The local `nvidia-lib` version remains **0.4.4**. Multiple Models now calls its synchronous `query()` and `stream_events()` APIs directly from worker threads; no library API change was required for v0.7.7.

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

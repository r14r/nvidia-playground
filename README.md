# NVIDIA NIM Playground

Version **0.7.3**

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
Runs the same prompt against all selected models concurrently. Each model is executed as its own `asyncio.Task`; provider calls reuse the same synchronous OpenAI/NVIDIA request path as Single model.

The first result tab is **Status**, followed by one tab per model. Each model tab contains **Response** and **Inspector**.

## Multi-model live rendering

The asyncio event loop runs in a dedicated background thread while Streamlit rendering stays on the main script thread. Every streaming content delta is forwarded immediately and updates the corresponding model's **Response** tab while the remaining model tasks continue running.

The default **Max Tokens** generation budget is **16384**.

`nvidia-lib` version **0.4.3** uses the same provider transport for Single and Multiple models.

## Runtime settings

The Run sidebar contains Model / Models, Temperature, Top P, Max Tokens, Streaming, and Raw chunk data.

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

## Model catalog

```bash
just models
```

executes:

```bash
nvidia-cli models --list --details --with-api-key --json --save models.json
```

`models.json` and `.env` are intentionally excluded from Git.

## Multiple-model provider transport fix

Single model and Multiple models now use the same NVIDIA/OpenAI provider request path while Multiple models keeps one `asyncio.Task` per selected model.

Blocking requests call synchronous `query()` through `asyncio.to_thread()`. Streaming requests consume synchronous `stream_events()` in one worker thread per model and forward every event immediately back into the async task.

This removes the separate asynchronous provider transport that caused first-word-only responses while preserving concurrent execution and live Response-tab updates.

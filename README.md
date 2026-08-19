# NVIDIA NIM Playground

Version **0.7.4**

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

Version 0.7.2 moves the asyncio event loop into a dedicated background thread while Streamlit rendering stays on the main script thread. Every streaming content delta is forwarded immediately and updates the corresponding model's **Response** tab while the remaining model tasks continue running.

The default **Max Tokens** generation budget is **2048** for broad compatibility across NVIDIA NIM models. Sessions carrying the v0.7.2 application default of 16384 are migrated once back to 2048.

`nvidia-lib` version **0.4.4** uses the same provider transport for Single and Multiple models and also normalizes OpenAI-compatible list-based content blocks into one complete response string.

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


## Multiple-model provider transport fix

Single model and Multiple models now use the same NVIDIA/OpenAI provider
request path while Multiple models keeps one `asyncio.Task` per selected model.

Blocking requests call synchronous `query()` through `asyncio.to_thread()`.
Streaming requests consume synchronous `stream_events()` in one worker thread
per model and forward every event immediately back into the async task.

This removes the separate asynchronous provider transport that caused
first-word-only responses while preserving concurrent execution and live
Response-tab updates.


## v0.7.4 application helpers and console lifecycle

Shared Streamlit helpers moved to `app/lib/` and are imported through `app.lib.*` so they do not collide with the separate `lib.nvidia` package.

Default User Prompt:

```text
**Describe advanced Streamlit features in three sentences per feature. No further information or additional details.**
```

The application default for `Max Tokens` is restored to 2048 for broad model compatibility.

Console output for every request follows this lifecycle:

```text
2026-08-19T19:09:30.614+02:00 Run Prompt on Model: meta/llama-3.3-70b-instruct
2026-08-19T19:09:30.614+02:00 System Prompt: "You are a helpful technical assistant."
2026-08-19T19:09:30.614+02:00 User Prompt: "**Describe advanced Streamlit features in three sentences per feature. No further information or additional details.**"
2026-08-19T19:09:30.614+02:00  Connect to NVIDIA
2026-08-19T19:09:30.614+02:00  Run Prompt
2026-08-19T19:09:30.614+02:00  Waiting for Response
2026-08-19T19:09:30.614+02:00  Response:
```

The completed response is printed after the `Response:` line. Streaming chunks are not printed individually.

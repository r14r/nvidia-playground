# NVIDIA NIM Playground

Version **0.6.3**

## Navigation

```text
Run

Models
  ├── Info
  ├── Metadata
  └── Catalog
```

The separate **Settings** page has been removed.

## Run sidebar

The Run page sidebar now contains all runtime settings:

- Model
- Temperature
- Top P
- Max Tokens
- Streaming
- Raw chunk data

The model selected in the sidebar is passed directly to the NVIDIA request:

```python
nvidia.query(..., model=selected_model)
```

or:

```python
nvidia.stream_events(..., model=selected_model)
```

## Default model

**Models → Catalog** contains a `Default model` selector and
`Save default model` button.

Saving updates `models.json` so exactly one model contains:

```json
"default": true
```

The file is written atomically by `nvidia-lib`.

The selected default is also used as the current Run-page selection.

## nvidia-lib

Library version: **0.3.0**

New API:

```python
from lib.nvidia import NVIDIA

nvidia = NVIDIA()

nvidia.set_default_model("minimax")

result = nvidia.query("why is the sky blue")
```

Without an explicit `model=`, `query()` uses the model marked as default in
`models.json`.

## Run

```bash
just run
```

## Library

```bash
just build-lib
just install-lib
```

## Model catalog

```bash
just models
```

executes:

```bash
nvidia-cli models --list --details --with-api-key --json --save models.json
```


## Run view

The Run page does not display the selected model as a separate information
block. The model is selected only through the Run sidebar and is used directly
for prompt execution.

Prompt editing is split into two tabs:

- **System Prompt**
- **User Prompt**

## Sidebar tooltips

The Run sidebar explains the two streaming-related switches directly in the UI:

- **Streaming**: receives and renders model output incrementally as chunks arrive.
- **Raw chunk data**: keeps the original low-level chunk payloads for debugging and inspection.



## Live streaming

In streaming mode the Run page updates both the response and the Stream Inspector immediately for every received event. The response uses an explicit Streamlit placeholder instead of waiting for post-stream rendering.


## Run modes

The Run navigation now contains:

```text
Run
  ├── Single model
  └── Multiple models
```

### Single model

Runs the current prompt against one selected model. Results are shown in two
tabs:

- **Response** — live model output
- **Inspector** — streaming chunks, gap timing, and optional raw chunks

### Multiple models

Select multiple models in the sidebar and run the same prompt against all of
them concurrently. Requests are executed with a thread pool. A result tab is
created for each selected model; each model tab contains its own **Response**
and **Inspector** tabs.

Streaming events are collected in worker threads and rendered by the Streamlit
main thread so live updates remain safe.


## update-cli

Project setup is configured in `update-cli.yaml`.

Run setup:

```bash
just setup-update-cli
# update-cli setup manifest ./update-cli.yaml
```

Start the Streamlit application through update-cli:

```bash
just run-update-cli
# update-cli setup manifest ./update-cli.yaml --setup-task run
```

The `run` task executes:

```bash
.venv/bin/streamlit run app/app.py
```

## Multiple-model tab identity

Multiple-model results use explicit Streamlit tab keys. Each model's nested
**Response** / **Inspector** tab group receives its own unique key to prevent
`StreamlitDuplicateElementId` when two or more models are selected.

This requires Streamlit 1.55 or newer.

## Console execution logging

Every Single model and Multiple models run writes its execution steps to the
console running Streamlit.

Normal log lines contain only operational metadata such as:

- selected model
- streaming/blocking mode
- request start and completion
- first-event notification
- chunk count
- character count
- TTFT and total duration
- result collection and payload storage

The console deliberately does **not** print chunk content or response text for
successful requests.

If a request fails, error diagnostics include the exception and, when
available, the partial response, chunk log, and raw chunks received before the
failure.


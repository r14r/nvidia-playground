# NVIDIA NIM Playground

Version **0.6.2**

Streamlit playground for NVIDIA NIM models with live streaming, chunk inspection, single-model and parallel multi-model execution.

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

## Run · Single model

Select one model in the sidebar and run the prompt against it.

Runtime settings:

- Model
- Temperature
- Top P
- Max Tokens
- Streaming
- Raw chunk data

Prompt editing uses two tabs:

- **System Prompt**
- **User Prompt**

Results use two tabs:

- **Response** — live streamed model output
- **Inspector** — streaming chunks, timing and optional raw chunk payloads

## Run · Multiple models

Select multiple models and execute the same prompt against all of them in parallel.

Each selected model gets its own result tab. Inside that model tab are **Response** and **Inspector** tabs.

The requests run concurrently through `ThreadPoolExecutor`. Streaming events are collected by worker threads and rendered by the Streamlit main thread.

### Streamlit tab identity

Multiple model tabs use explicit unique keys for every nested `st.tabs()` group. This prevents `StreamlitDuplicateElementId` when two or more models are selected.

Streamlit **1.55 or newer** is required.

## Models · Catalog

The Catalog page can change the default model in `models.json`. Saving updates the catalog so exactly one model has:

```json
"default": true
```

`models.json` contains credentials and is excluded from Git.

## nvidia-lib

Library version: **0.3.0**

```python
from lib.nvidia import NVIDIA

nvidia = NVIDIA()
nvidia.set_default_model("minimax")
result = nvidia.query("why is the sky blue")
```

## Setup

```bash
just setup
```

## Run

```bash
just run
```

## update-cli

The project manifest is named:

```text
update-cli.yaml
```

Run the setup workflow:

```bash
just setup-update-cli
# update-cli setup manifest ./update-cli.yaml
```

Start Streamlit through the manifest `run` task:

```bash
just run-update-cli
# update-cli setup manifest ./update-cli.yaml --setup-task run
```

The task executes:

```bash
.venv/bin/streamlit run app/app.py
```

## Model catalog refresh

```bash
just models
```

This executes:

```bash
nvidia-cli models --list --details --with-api-key --json --save models.json
```

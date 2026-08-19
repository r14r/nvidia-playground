# NVIDIA NIM Playground

Version **0.4.2**

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

The main Run view displays only the currently selected model name. Runtime parameters remain in the sidebar.

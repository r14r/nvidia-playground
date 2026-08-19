default:
    @just --list

# Full setup: env, venv, dependencies, model catalog, compile and tests
setup:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ ! -f .env ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        echo ".env already exists."
    fi

    if [ ! -x .venv/bin/python ]; then
        python3 -m venv .venv
        echo "Created Python virtual environment."
    else
        echo "Python virtual environment already exists."
    fi

    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt

    just models
    just compile
    just test

# Execute setup.yaml through update-cli
setup-update-cli:
    update-cli --setup

# Create optional .env from .env.example only when missing
env:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f .env ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        echo ".env already exists; unchanged."
    fi

# Install/update app dependencies and editable local NVIDIA library
install:
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt

# Build nvidia-lib as wheel and source distribution under lib/dist/
build-lib:
    #!/usr/bin/env bash
    set -euo pipefail

    if [ ! -x .venv/bin/python ]; then
        echo "ERROR: .venv missing. Run 'just setup' or 'just install' first." >&2
        exit 1
    fi

    rm -rf lib/dist lib/build
    find lib -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +

    .venv/bin/python -m build --outdir lib/dist lib

    echo
    echo "Built nvidia-lib packages:"
    ls -lh lib/dist/

# Build and install nvidia-lib from the generated wheel
install-lib: build-lib
    #!/usr/bin/env bash
    set -euo pipefail

    wheel="$(find lib/dist -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"

    if [ -z "$wheel" ]; then
        echo "ERROR: no wheel found in lib/dist" >&2
        exit 1
    fi

    echo "Installing $wheel"
    .venv/bin/python -m pip install --force-reinstall "$wheel"

    echo
    .venv/bin/python -m pip show nvidia-lib

# Refresh NVIDIA model catalog with details and API keys
models:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v nvidia-cli >/dev/null 2>&1; then
        echo "ERROR: nvidia-cli was not found in PATH." >&2
        exit 1
    fi
    nvidia-cli models --list --details --with-api-key --json --save models.json
    test -s models.json
    python3 -m json.tool models.json >/dev/null
    echo "Saved NVIDIA models, endpoints and credentials to models.json"

# Refresh catalog and print safe metadata for MODEL; partial names supported
modelinfo model="": models
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -x .venv/bin/python ]; then
        echo "ERROR: .venv missing. Run 'just setup' first." >&2
        exit 1
    fi
    if [ -n "{{model}}" ]; then
        .venv/bin/python -m lib.nvidia.cli --file models.json --model "{{model}}"
    else
        .venv/bin/python -m lib.nvidia.cli --file models.json
    fi

# Compile app and local NVIDIA library
compile:
    .venv/bin/python -m compileall -q app lib/src/lib/nvidia
    @echo "Python compile OK"

# Run NVIDIA library unit tests
test:
    .venv/bin/python -m unittest discover -s lib/tests -v

# Start Streamlit app
run:
    .venv/bin/streamlit run app/app.py

# Start Streamlit without browser auto-open
serve:
    .venv/bin/streamlit run app/app.py --server.headless true

# Compile, test and validate models.json when present
check: compile test
    @if [ -f models.json ]; then python3 -m json.tool models.json >/dev/null && echo "models.json OK"; fi

# Show installed package versions
versions:
    .venv/bin/python --version
    .venv/bin/pip show nvidia-lib streamlit openai python-dotenv pandas watchdog build

# Remove generated local files
clean:
    rm -rf app/__pycache__ .streamlit lib/build lib/dist
    find app lib -name '__pycache__' -type d -prune -exec rm -rf {} +
    find app lib -name '*.pyc' -delete
    find lib -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +

# Remove virtual environment, optional .env and credential-bearing model catalog
distclean: clean
    rm -rf .venv models.json .env

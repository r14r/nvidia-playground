from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from app.lib.console import connect_to_provider as console_connect
from app.lib.console import error as console_error
from app.lib.console import execute_prompt as console_execute
from app.lib.console import response as console_response
from app.lib.console import run_prompt as console_run_prompt
from app.lib.console import waiting_for_response as console_waiting
from app.lib.providers import (
    PROVIDERS,
    console_provider_name,
    provider_can_run,
    provider_query,
    provider_stream_events,
    provider_supports_streaming,
    runtime_model_ids,
)
from app.lib.run_common import (
    json_bytes,
    render_prompt_tabs,
    render_runtime_settings,
    response_filename,
)
from app.lib.shared import (
    APP_VERSION,
    ensure_base_settings,
    require_nvidia,
)


def classify_request_error(
    exc: Exception,
    provider: str,
) -> dict[str, str]:
    raw_error = str(exc)
    normalized = raw_error.lower()

    if "degraded" in normalized and "cannot be invoked" in normalized:
        provider_name = console_provider_name(provider)
        return {
            "type": "provider_unavailable",
            "status": "Unavailable",
            "message": (
                f"{provider_name} backend for this model is temporarily "
                "unavailable (DEGRADED). Retry later or choose another model."
            ),
            "raw": raw_error,
        }

    return {
        "type": "request_error",
        "status": "Error",
        "message": raw_error,
        "raw": raw_error,
    }


st.title("Multiple models")
st.caption(
    "Run the same prompt on multiple models in parallel worker threads "
    "or sequentially."
)

ensure_base_settings()

with st.sidebar:
    st.subheader("Run Settings")

    provider = st.selectbox(
        "Provider",
        options=PROVIDERS,
        key="run_provider",
        help=(
            "Ollama runs against localhost:11434. NVIDIA and models.json "
            "use the NVIDIA endpoints and credentials configured in models.json."
        ),
    )

nvidia = None
try:
    if provider in {"NVIDIA", "models.json"}:
        nvidia = require_nvidia()

    ids = runtime_model_ids(
        provider,
        nvidia=nvidia,
    )
except Exception as exc:
    st.error(f"Could not load models for {provider}: {exc}")
    st.stop()

if not ids:
    st.warning(f"No models are available from {provider}.")
    st.stop()

current_selection = [
    model
    for model in st.session_state.get("selected_models", [])
    if model in ids
]
if not current_selection:
    preferred = st.session_state.get("selected_model")
    current_selection = [preferred] if preferred in ids else ids[:1]
st.session_state["selected_models"] = current_selection

with st.sidebar:
    selected_models = st.multiselect(
        "Models",
        options=ids,
        key="selected_models",
        help="Select multiple models to run the same prompt.",
    )

    selected_supports_streaming = all(
        provider_supports_streaming(
            provider,
            model,
            nvidia=nvidia,
        )
        for model in selected_models
    )
    render_runtime_settings(
        supports_streaming=(
            selected_supports_streaming if selected_models else True
        )
    )

    run_parallel = st.toggle(
        "Run Parallel",
        value=True,
        key="run_parallel",
        help=(
            "Run selected models concurrently in separate worker threads. "
            "No asyncio execution is used."
        ),
    )

render_prompt_tabs()

runnable_models = [
    model
    for model in selected_models
    if provider_can_run(provider, model, nvidia=nvidia)
]
blocked_models = [
    model for model in selected_models if model not in runnable_models
]

if blocked_models:
    if provider == "Ollama":
        st.warning(
            "These selected Ollama models are not available: "
            f"{', '.join(blocked_models)}"
        )
    else:
        st.warning(
            "These selected NVIDIA models cannot run because base_url or "
            f"api_key is missing: {', '.join(blocked_models)}"
        )

run = st.button(
    "Run Prompt on selected models",
    type="primary",
    width="stretch",
    disabled=not runnable_models,
)

if run:
    prompt = st.session_state["prompt"]
    system_prompt = st.session_state["system_prompt"]
    temperature = float(st.session_state["temperature"])
    top_p = float(st.session_state["top_p"])
    max_tokens = int(st.session_state["max_tokens"])
    streaming = bool(st.session_state["streaming"])
    show_raw_chunks = bool(st.session_state["show_raw_chunks"])
    run_parallel = bool(st.session_state["run_parallel"])

    result_tabs = st.tabs(
        ["Status", *runnable_models],
        key="multiple_model_results",
    )
    status_tab = result_tabs[0]
    model_tabs = result_tabs[1:]

    status_rows: dict[str, dict[str, Any]] = {
        model: {
            "Provider": provider,
            "Model": model,
            "Status": "Queued",
            "Execution": "Parallel" if run_parallel else "Sequential",
            "Mode": (
                "Streaming"
                if streaming
                and provider_supports_streaming(
                    provider,
                    model,
                    nvidia=nvidia,
                )
                else "Blocking"
            ),
            "Elapsed": "—",
            "TTFT": "—",
            "Chunks": 0,
            "Characters": 0,
            "Error": "",
        }
        for model in runnable_models
    }

    with status_tab:
        st.subheader("Execution status")
        status_summary = st.empty()
        status_placeholder = st.empty()

    ui: dict[str, dict[str, Any]] = {}
    for model_index, (model, tab) in enumerate(
        zip(runnable_models, model_tabs)
    ):
        with tab:
            status = st.empty()
            response_tab, inspector_tab = st.tabs(
                ["Response", "Inspector"],
                key=f"model_result_tabs_{model_index}",
            )

            with response_tab:
                response = st.empty()
                metrics = st.empty()

            with inspector_tab:
                inspector = st.empty()
                inspector_stats = st.empty()
                raw = st.empty()

            status.info("Queued…")
            ui[model] = {
                "status": status,
                "response": response,
                "metrics": metrics,
                "inspector": inspector,
                "inspector_stats": inspector_stats,
                "raw": raw,
            }

    def render_status() -> None:
        rows = [status_rows[model] for model in runnable_models]
        completed = sum(
            row["Status"] in {"Completed", "Error", "Unavailable"}
            for row in rows
        )
        running = sum(row["Status"] == "Running" for row in rows)
        errors = sum(row["Status"] == "Error" for row in rows)
        unavailable = sum(
            row["Status"] == "Unavailable" for row in rows
        )
        execution = "Parallel" if run_parallel else "Sequential"

        status_summary.caption(
            f"Provider: {provider} · "
            f"Execution: {execution} · "
            f"Completed: {completed}/{len(rows)} · "
            f"Running: {running} · "
            f"Unavailable: {unavailable} · "
            f"Errors: {errors}"
        )
        status_placeholder.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

    render_status()

    event_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def run_model_sync(model: str) -> dict[str, Any]:
        started = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        raw_chunks: list[dict[str, Any]] = []
        text = ""
        ttft_ms = None
        use_streaming = (
            streaming
            and provider_supports_streaming(
                provider,
                model,
                nvidia=nvidia,
            )
        )

        console_run_prompt(
            model,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
        console_connect(console_provider_name(provider))
        console_execute()
        console_waiting()

        event_queue.put(
            {
                "type": "started",
                "model": model,
                "mode": "Streaming" if use_streaming else "Blocking",
            }
        )

        try:
            if use_streaming:
                for event in provider_stream_events(
                    provider,
                    prompt,
                    model=model,
                    nvidia=nvidia,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    include_raw=show_raw_chunks,
                ):
                    content = event.get("content", "")
                    if content and ttft_ms is None:
                        ttft_ms = float(event["elapsed_ms"])

                    row = {
                        "chunk": event["chunk"],
                        "time_ms": event["elapsed_ms"],
                        "gap_ms": event["gap_ms"],
                        "chars": event["chars"],
                        "delta": content,
                        "finish_reason": event.get("finish_reason"),
                    }
                    chunks.append(row)

                    if show_raw_chunks and "raw" in event:
                        raw_chunks.append(event["raw"])

                    if content:
                        text += content

                    event_queue.put(
                        {
                            "type": "chunk",
                            "model": model,
                            "text": text,
                            "chunks": list(chunks),
                            "raw_chunks": list(raw_chunks),
                            "ttft_ms": ttft_ms,
                            "elapsed": time.perf_counter() - started,
                        }
                    )
            else:
                text = provider_query(
                    provider,
                    prompt,
                    model=model,
                    nvidia=nvidia,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

            result = {
                "provider": provider,
                "model": model,
                "text": text,
                "chunks": chunks,
                "raw_chunks": raw_chunks,
                "ttft_ms": ttft_ms,
                "total_time_seconds": time.perf_counter() - started,
                "error": None,
                "error_type": None,
                "provider_error": None,
            }
            console_response(text)
        except Exception as exc:
            console_error(model, exc, partial_response=text)
            error_info = classify_request_error(exc, provider)
            result = {
                "provider": provider,
                "model": model,
                "text": text,
                "chunks": chunks,
                "raw_chunks": raw_chunks,
                "ttft_ms": ttft_ms,
                "total_time_seconds": time.perf_counter() - started,
                "error": error_info["message"],
                "error_type": error_info["type"],
                "provider_error": error_info["raw"],
            }

        event_queue.put(
            {
                "type": "completed",
                "model": model,
                "result": result,
            }
        )
        return result

    def producer() -> None:
        try:
            if run_parallel:
                with ThreadPoolExecutor(
                    max_workers=len(runnable_models),
                    thread_name_prefix="provider-model",
                ) as executor:
                    futures = [
                        executor.submit(run_model_sync, model)
                        for model in runnable_models
                    ]
                    for future in as_completed(futures):
                        future.result()
            else:
                for model in runnable_models:
                    run_model_sync(model)
        except Exception as exc:
            event_queue.put(
                {
                    "type": "runner_error",
                    "model": "",
                    "error": str(exc),
                }
            )
        finally:
            event_queue.put({"type": "runner_done", "model": ""})

    runner_thread = threading.Thread(
        target=producer,
        name="multi-model-runner",
        daemon=True,
    )
    runner_thread.start()

    results: dict[str, dict[str, Any]] = {}
    runner_done = False

    while not runner_done or len(results) < len(runnable_models):
        try:
            event = event_queue.get(timeout=0.05)
        except queue.Empty:
            time.sleep(0.01)
            continue

        event_type = event["type"]

        if event_type == "runner_done":
            runner_done = True
            continue

        if event_type == "runner_error":
            st.error(f"Model runner failed: {event['error']}")
            runner_done = True
            break

        model = event["model"]
        model_ui = ui[model]
        row = status_rows[model]

        if event_type == "started":
            row["Status"] = "Running"
            row["Mode"] = event["mode"]
            model_ui["status"].info("Running…")
            render_status()
            continue

        if event_type == "chunk":
            chunks = event["chunks"]
            response_text = event["text"]

            row["Status"] = "Running"
            row["Elapsed"] = f"{event['elapsed']:.2f} s"
            row["TTFT"] = (
                f"{event['ttft_ms']:.0f} ms"
                if event["ttft_ms"] is not None
                else "—"
            )
            row["Chunks"] = len(chunks)
            row["Characters"] = len(response_text)

            model_ui["response"].markdown(response_text + "▌")

            if chunks:
                model_ui["inspector"].dataframe(
                    pd.DataFrame(chunks),
                    width="stretch",
                    hide_index=True,
                    height=430,
                )
                gaps = [
                    chunk["gap_ms"]
                    for chunk in chunks
                    if chunk["chunk"] > 1
                ]
                if gaps:
                    model_ui["inspector_stats"].caption(
                        f"Avg. gap: {sum(gaps) / len(gaps):.1f} ms · "
                        f"Max gap: {max(gaps):.1f} ms"
                    )
            render_status()
            continue

        if event_type == "completed":
            result = event["result"]
            results[model] = result

            if result["error_type"] == "provider_unavailable":
                row["Status"] = "Unavailable"
            elif result["error"]:
                row["Status"] = "Error"
            else:
                row["Status"] = "Completed"

            row["Elapsed"] = f"{result['total_time_seconds']:.2f} s"
            row["TTFT"] = (
                f"{result['ttft_ms']:.0f} ms"
                if result["ttft_ms"] is not None
                else "—"
            )
            row["Chunks"] = len(result["chunks"])
            row["Characters"] = len(result["text"])
            row["Error"] = result["error"] or ""

            if result["error_type"] == "provider_unavailable":
                model_ui["status"].warning(
                    f"{model}: {result['error']}"
                )
            elif result["error"]:
                model_ui["status"].error(
                    f"{model}: {result['error']}"
                )
            else:
                model_ui["status"].success("Completed")

            if result["error_type"] == "provider_unavailable":
                model_ui["response"].warning(result["error"])
            else:
                model_ui["response"].markdown(result["text"])

            with model_ui["metrics"].container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("TTFT", row["TTFT"])
                m2.metric("Total Time", row["Elapsed"])
                m3.metric("Chunks", row["Chunks"])
                m4.metric("Characters", row["Characters"])

            if not streaming or row["Mode"] == "Blocking":
                model_ui["inspector"].info(
                    "Streaming is disabled for this run."
                )
            elif not result["chunks"]:
                model_ui["inspector"].info(
                    "No streaming chunks received."
                )

            if result["provider_error"]:
                model_ui["inspector_stats"].caption(
                    f"Provider error: {result['provider_error']}"
                )

            if show_raw_chunks and result["raw_chunks"]:
                with model_ui["raw"].container():
                    with st.expander("Raw chunks"):
                        st.json(result["raw_chunks"])

            render_status()

    runner_thread.join(timeout=1.0)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "application": {
            "name": "nvidia-playground",
            "version": APP_VERSION,
        },
        "request": {
            "provider": provider,
            "models": runnable_models,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "streaming": streaming,
            "run_parallel": run_parallel,
        },
        "results": results,
    }
    st.session_state["last_multi_response_payload"] = payload

if "last_multi_response_payload" in st.session_state:
    st.divider()
    st.download_button(
        "Save all responses as JSON",
        data=json_bytes(st.session_state["last_multi_response_payload"]),
        file_name=response_filename("multi-model-response"),
        mime="application/json",
        width="stretch",
    )

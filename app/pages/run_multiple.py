from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from console import error as console_error
from console import header as console_header
from console import step as console_step

from run_common import (
    json_bytes,
    render_prompt_tabs,
    render_runtime_settings,
    response_filename,
)
from shared import APP_VERSION, model_ids, require_nvidia

st.title("Multiple models")
st.caption("Run the same prompt against multiple NVIDIA NIM models in parallel.")

nvidia = require_nvidia()
ids = model_ids(nvidia)

with st.sidebar:
    st.subheader("Run Settings")

    default_model = st.session_state.get("selected_model")
    default_selection = [default_model] if default_model in ids else ids[:1]

    selected_models = st.multiselect(
        "Models",
        options=ids,
        default=default_selection,
        key="selected_models",
        help="Select two or more models to run the same prompt in parallel.",
    )

    render_runtime_settings(supports_streaming=True)

render_prompt_tabs()

runnable_models = [
    model for model in selected_models if nvidia.can_run(model)
]
blocked_models = [
    model for model in selected_models if model not in runnable_models
]

if blocked_models:
    st.warning(
        "These selected models cannot run because base_url or api_key is "
        f"missing: {', '.join(blocked_models)}"
    )

run = st.button(
    "Run Prompt on selected models",
    type="primary",
    width="stretch",
    disabled=not runnable_models,
)


if run:
    console_header("multi", "MULTI-MODEL PROMPT RUN")
    console_step(
        "multi",
        1,
        "Run started",
        models=len(runnable_models),
        streaming=bool(st.session_state["streaming"]),
    )
    console_step(
        "multi",
        2,
        "Model selection validated",
        runnable=len(runnable_models),
        blocked=len(blocked_models),
    )

    # Snapshot all Streamlit state before starting worker threads. Worker
    # threads do not call Streamlit APIs.
    prompt = st.session_state["prompt"]
    system_prompt = st.session_state["system_prompt"]
    temperature = float(st.session_state["temperature"])
    top_p = float(st.session_state["top_p"])
    max_tokens = int(st.session_state["max_tokens"])
    streaming = bool(st.session_state["streaming"])
    show_raw_chunks = bool(st.session_state["show_raw_chunks"])

    console_step("multi", 3, "Request parameters snapshotted")

    # Rebind worker to immutable request values instead of reading Streamlit
    # session state from worker threads.
    def run_model(model: str, event_queue: queue.Queue) -> dict[str, Any]:
        started = time.perf_counter()
        console_step("multi", 4, "Model worker started", model=model)
        chunks: list[dict] = []
        raw_chunks: list[dict] = []
        text = ""
        ttft_ms = None

        try:
            if streaming:
                console_step(
                    "multi",
                    5,
                    "Sending streaming request",
                    model=model,
                )
                first_event_logged = False

                for event in nvidia.stream_events(
                    prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    include_raw=show_raw_chunks,
                ):
                    content = event.get("content", "")

                    if not first_event_logged:
                        console_step(
                            "multi",
                            6,
                            "First stream event received",
                            model=model,
                        )
                        first_event_logged = True

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
                        }
                    )

                console_step(
                    "multi",
                    7,
                    "Streaming request finished",
                    model=model,
                    chunks=len(chunks),
                    characters=len(text),
                )
            else:
                console_step(
                    "multi",
                    5,
                    "Sending blocking request",
                    model=model,
                )
                text = nvidia.query(
                    prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                console_step(
                    "multi",
                    6,
                    "Blocking request finished",
                    model=model,
                    characters=len(text),
                )

            console_step(
                "multi",
                8,
                "Model worker completed",
                model=model,
                duration=f"{time.perf_counter() - started:.3f}s",
            )

            return {
                "model": model,
                "text": text,
                "chunks": chunks,
                "raw_chunks": raw_chunks,
                "ttft_ms": ttft_ms,
                "total_time_seconds": time.perf_counter() - started,
                "error": None,
            }
        except Exception as exc:
            console_error(
                "multi",
                f"Model request failed ({model})",
                exc=exc,
                response=text,
                chunks=chunks,
                raw_chunks=raw_chunks,
            )
            return {
                "model": model,
                "text": text,
                "chunks": chunks,
                "raw_chunks": raw_chunks,
                "ttft_ms": ttft_ms,
                "total_time_seconds": time.perf_counter() - started,
                "error": str(exc),
            }

    console_step("multi", 9, "Creating result tabs")

    result_tabs = st.tabs(
        runnable_models,
        key="multiple_model_results",
    )
    ui: dict[str, dict[str, Any]] = {}

    for model_index, (model, tab) in enumerate(
        zip(runnable_models, result_tabs)
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

            status.info("Running…")
            ui[model] = {
                "status": status,
                "response": response,
                "metrics": metrics,
                "inspector": inspector,
                "inspector_stats": inspector_stats,
                "raw": raw,
            }

    events: queue.Queue = queue.Queue()
    results: dict[str, dict[str, Any]] = {}

    console_step(
        "multi",
        10,
        "Starting parallel model execution",
        workers=len(runnable_models),
    )

    with ThreadPoolExecutor(max_workers=len(runnable_models)) as executor:
        futures = {
            model: executor.submit(run_model, model, events)
            for model in runnable_models
        }

        while len(results) < len(futures):
            # Render every queued chunk on the Streamlit main thread.
            try:
                while True:
                    event = events.get_nowait()
                    model = event["model"]
                    model_ui = ui[model]

                    model_ui["response"].markdown(event["text"] + "▌")

                    chunks = event["chunks"]
                    if chunks:
                        model_ui["inspector"].dataframe(
                            pd.DataFrame(chunks),
                            width="stretch",
                            hide_index=True,
                            height=430,
                        )
                        gaps = [
                            row["gap_ms"]
                            for row in chunks
                            if row["chunk"] > 1
                        ]
                        if gaps:
                            model_ui["inspector_stats"].caption(
                                f"Avg. gap: {sum(gaps) / len(gaps):.1f} ms · "
                                f"Max gap: {max(gaps):.1f} ms"
                            )
            except queue.Empty:
                pass

            for model, future in futures.items():
                if model not in results and future.done():
                    result = future.result()
                    results[model] = result
                    model_ui = ui[model]

                    console_step(
                        "multi",
                        11,
                        "Model result collected",
                        model=model,
                        error=bool(result["error"]),
                    )

                    if result["error"]:
                        model_ui["status"].error(result["error"])
                    else:
                        model_ui["status"].success("Completed")

                    model_ui["response"].markdown(result["text"])

                    with model_ui["metrics"].container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(
                            "TTFT",
                            f"{result['ttft_ms']:.0f} ms"
                            if result["ttft_ms"] is not None
                            else "—",
                        )
                        m2.metric(
                            "Total Time",
                            f"{result['total_time_seconds']:.2f} s",
                        )
                        m3.metric("Chunks", len(result["chunks"]))
                        m4.metric("Characters", len(result["text"]))

                    if not streaming:
                        model_ui["inspector"].info("Streaming is disabled.")
                    elif not result["chunks"]:
                        model_ui["inspector"].info(
                            "No streaming chunks received."
                        )

                    if show_raw_chunks and result["raw_chunks"]:
                        with model_ui["raw"].container():
                            with st.expander("Raw chunks"):
                                st.json(result["raw_chunks"])

            if len(results) < len(futures):
                time.sleep(0.03)

    console_step(
        "multi",
        12,
        "All model results collected",
        results=len(results),
    )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "application": {
            "name": "nvidia-playground",
            "version": APP_VERSION,
        },
        "request": {
            "models": runnable_models,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "streaming": streaming,
        },
        "results": results,
    }
    st.session_state["last_multi_response_payload"] = payload
    console_step("multi", 13, "Combined response payload stored")
    console_step("multi", 14, "Run completed")

if "last_multi_response_payload" in st.session_state:
    st.divider()
    st.download_button(
        "Save all responses as JSON",
        data=json_bytes(st.session_state["last_multi_response_payload"]),
        file_name=response_filename("nvidia-multi-response"),
        mime="application/json",
        width="stretch",
    )

from __future__ import annotations

import asyncio
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from console import error as console_error
from console import result as console_result
from console import run_prompt as console_run_prompt
from console import selected_model as console_selected_model
from run_common import (
    json_bytes,
    render_prompt_tabs,
    render_runtime_settings,
    response_filename,
)
from shared import APP_VERSION, model_ids, require_nvidia

st.title("Multiple models")
st.caption("Run the same prompt concurrently with native asyncio tasks.")

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
        help="Select multiple models to run the same prompt concurrently.",
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


def model_supports_streaming(model: str) -> bool:
    info = nvidia.model(model, safe=True)
    capabilities = info.get("capabilities") or {}
    return capabilities.get("streaming", True) is not False


if run:
    prompt = st.session_state["prompt"]
    system_prompt = st.session_state["system_prompt"]
    temperature = float(st.session_state["temperature"])
    top_p = float(st.session_state["top_p"])
    max_tokens = int(st.session_state["max_tokens"])
    streaming = bool(st.session_state["streaming"])
    show_raw_chunks = bool(st.session_state["show_raw_chunks"])

    result_tabs = st.tabs(
        ["Status", *runnable_models],
        key="multiple_model_results",
    )
    status_tab = result_tabs[0]
    model_tabs = result_tabs[1:]

    status_rows: dict[str, dict[str, Any]] = {
        model: {
            "Model": model,
            "Status": "Queued",
            "Mode": (
                "Streaming"
                if streaming and model_supports_streaming(model)
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
            row["Status"] in {"Completed", "Error"}
            for row in rows
        )
        running = sum(row["Status"] == "Running" for row in rows)
        errors = sum(row["Status"] == "Error" for row in rows)
        status_summary.caption(
            f"Completed: {completed}/{len(rows)} · "
            f"Running: {running} · Errors: {errors}"
        )
        status_placeholder.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

    render_status()

    async def run_model(
        model: str,
        event_queue: asyncio.Queue[dict[str, Any]],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        raw_chunks: list[dict[str, Any]] = []
        text = ""
        ttft_ms = None
        use_streaming = streaming and model_supports_streaming(model)

        console_selected_model(model)
        console_run_prompt(
            model,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

        await event_queue.put(
            {
                "type": "started",
                "model": model,
                "mode": "Streaming" if use_streaming else "Blocking",
            }
        )

        try:
            if use_streaming:
                async for event in nvidia.async_stream_events(
                    prompt,
                    model=model,
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

                    await event_queue.put(
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
                text = await nvidia.async_query(
                    prompt,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

            result = {
                "model": model,
                "text": text,
                "chunks": chunks,
                "raw_chunks": raw_chunks,
                "ttft_ms": ttft_ms,
                "total_time_seconds": time.perf_counter() - started,
                "error": None,
            }
            console_result(model, text)
        except Exception as exc:
            console_error(model, exc, partial_response=text)
            result = {
                "model": model,
                "text": text,
                "chunks": chunks,
                "raw_chunks": raw_chunks,
                "ttft_ms": ttft_ms,
                "total_time_seconds": time.perf_counter() - started,
                "error": str(exc),
            }

        await event_queue.put(
            {
                "type": "completed",
                "model": model,
                "result": result,
            }
        )
        return result

    async def async_producer(
        event_queue: queue.Queue[dict[str, Any]],
    ) -> None:
        """Run one native asyncio task per model in a background event loop."""
        async def forward_model(model: str) -> dict[str, Any]:
            # run_model expects an awaitable queue.put(). Adapt the standard
            # thread-safe queue without moving Streamlit calls off the main
            # script thread.
            class EventQueueAdapter:
                async def put(self, event: dict[str, Any]) -> None:
                    event_queue.put(event)

            return await run_model(model, EventQueueAdapter())

        tasks = [
            asyncio.create_task(
                forward_model(model),
                name=f"nvidia-{model}",
            )
            for model in runnable_models
        ]
        await asyncio.gather(*tasks)

    event_queue: queue.Queue[dict[str, Any]] = queue.Queue()

    def async_runner() -> None:
        try:
            asyncio.run(async_producer(event_queue))
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
        target=async_runner,
        name="nvidia-multi-asyncio",
        daemon=True,
    )
    runner_thread.start()

    results: dict[str, dict[str, Any]] = {}
    runner_done = False

    # Streamlit rendering remains on the script thread. The background
    # asyncio event loop only performs provider I/O and emits events.
    while not runner_done or len(results) < len(runnable_models):
        try:
            event = event_queue.get(timeout=0.05)
        except queue.Empty:
            # Give Streamlit a chance to flush already-enqueued UI deltas
            # while provider requests continue asynchronously.
            time.sleep(0.01)
            continue

        event_type = event["type"]

        if event_type == "runner_done":
            runner_done = True
            continue

        if event_type == "runner_error":
            st.error(f"Async runner failed: {event['error']}")
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

            # Update the model's Response tab for every received streaming
            # delta, independently from all other running model tasks.
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

            row["Status"] = "Error" if result["error"] else "Completed"
            row["Elapsed"] = f"{result['total_time_seconds']:.2f} s"
            row["TTFT"] = (
                f"{result['ttft_ms']:.0f} ms"
                if result["ttft_ms"] is not None
                else "—"
            )
            row["Chunks"] = len(result["chunks"])
            row["Characters"] = len(result["text"])
            row["Error"] = result["error"] or ""

            if result["error"]:
                model_ui["status"].error(result["error"])
            else:
                model_ui["status"].success("Completed")

            # Final render removes the live cursor.
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

if "last_multi_response_payload" in st.session_state:
    st.divider()
    st.download_button(
        "Save all responses as JSON",
        data=json_bytes(st.session_state["last_multi_response_payload"]),
        file_name=response_filename("nvidia-multi-response"),
        mime="application/json",
        width="stretch",
    )

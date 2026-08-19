from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from run_common import (
    json_bytes,
    render_prompt_tabs,
    render_runtime_settings,
    response_filename,
)
from shared import APP_VERSION, model_ids, require_nvidia

st.title("Single model")
st.caption("Run one prompt against one NVIDIA NIM model.")

nvidia = require_nvidia()
ids = model_ids(nvidia)

with st.sidebar:
    st.subheader("Run Settings")

    selected_model = st.selectbox(
        "Model",
        options=ids,
        key="selected_model",
    )

    selected_info = nvidia.model(selected_model, safe=True)
    capabilities = selected_info.get("capabilities") or {}
    supports_streaming = capabilities.get("streaming", True) is not False
    render_runtime_settings(supports_streaming=supports_streaming)

can_run = nvidia.can_run(selected_model)

render_prompt_tabs()

run = st.button(
    "Run Prompt",
    type="primary",
    width="stretch",
    disabled=not can_run,
)

if not can_run:
    st.warning(
        "The selected model cannot run because its models.json entry "
        "does not contain a usable base_url and api_key."
    )

if run:
    response_tab, inspector_tab = st.tabs(["Response", "Inspector"])

    chunk_log: list[dict] = []
    raw_chunks: list[dict] = []
    full_response = ""
    started = time.perf_counter()
    first_content_ms = None

    with response_tab:
        response_placeholder = st.empty()
        response_metrics = st.empty()

    with inspector_tab:
        inspector_placeholder = st.empty()
        inspector_stats_placeholder = st.empty()
        raw_placeholder = st.empty()

    try:
        if st.session_state["streaming"]:
            for event in nvidia.stream_events(
                st.session_state["prompt"],
                model=selected_model,
                system_prompt=st.session_state["system_prompt"],
                temperature=float(st.session_state["temperature"]),
                top_p=float(st.session_state["top_p"]),
                max_tokens=int(st.session_state["max_tokens"]),
                include_raw=bool(st.session_state["show_raw_chunks"]),
            ):
                content = event.get("content", "")

                if content and first_content_ms is None:
                    first_content_ms = float(event["elapsed_ms"])

                chunk_log.append(
                    {
                        "chunk": event["chunk"],
                        "time_ms": event["elapsed_ms"],
                        "gap_ms": event["gap_ms"],
                        "chars": event["chars"],
                        "delta": content,
                        "finish_reason": event.get("finish_reason"),
                    }
                )

                if st.session_state["show_raw_chunks"] and "raw" in event:
                    raw_chunks.append(event["raw"])

                if content:
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")

                inspector_placeholder.dataframe(
                    pd.DataFrame(chunk_log),
                    width="stretch",
                    hide_index=True,
                    height=430,
                )

                gaps = [
                    row["gap_ms"]
                    for row in chunk_log
                    if row["chunk"] > 1
                ]
                if gaps:
                    inspector_stats_placeholder.caption(
                        f"Avg. gap: {sum(gaps) / len(gaps):.1f} ms · "
                        f"Max gap: {max(gaps):.1f} ms"
                    )

            response_placeholder.markdown(full_response)
        else:
            full_response = nvidia.query(
                st.session_state["prompt"],
                model=selected_model,
                system_prompt=st.session_state["system_prompt"],
                temperature=float(st.session_state["temperature"]),
                top_p=float(st.session_state["top_p"]),
                max_tokens=int(st.session_state["max_tokens"]),
            )
            response_placeholder.markdown(full_response)
            inspector_placeholder.info("Streaming is disabled.")

        total_time = time.perf_counter() - started

        with response_metrics.container():
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric(
                "TTFT",
                f"{first_content_ms:.0f} ms"
                if first_content_ms is not None
                else "—",
            )
            metric2.metric("Total Time", f"{total_time:.2f} s")
            metric3.metric("Chunks", len(chunk_log))
            metric4.metric("Characters", len(full_response))

        if st.session_state["streaming"] and not chunk_log:
            inspector_placeholder.info("No streaming chunks received.")

        if st.session_state["show_raw_chunks"] and raw_chunks:
            with raw_placeholder.container():
                with st.expander("Raw chunks"):
                    st.json(raw_chunks)

        export_info = dict(selected_info)
        export_info.pop("api_key", None)

        st.session_state["last_response_payload"] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "application": {
                "name": "nvidia-playground",
                "version": APP_VERSION,
            },
            "model": selected_model,
            "model_info": export_info,
            "request": {
                "system_prompt": st.session_state["system_prompt"],
                "prompt": st.session_state["prompt"],
                "temperature": float(st.session_state["temperature"]),
                "top_p": float(st.session_state["top_p"]),
                "max_tokens": int(st.session_state["max_tokens"]),
                "streaming": bool(st.session_state["streaming"]),
            },
            "response": {
                "text": full_response,
                "characters": len(full_response),
            },
            "metrics": {
                "ttft_ms": first_content_ms,
                "total_time_seconds": round(total_time, 4),
                "chunks": len(chunk_log),
            },
            "chunks": chunk_log,
        }

        if st.session_state["show_raw_chunks"]:
            st.session_state["last_response_payload"]["raw_chunks"] = raw_chunks

    except Exception as exc:
        st.exception(exc)

if "last_response_payload" in st.session_state:
    st.divider()
    st.download_button(
        "Save response as JSON",
        data=json_bytes(st.session_state["last_response_payload"]),
        file_name=response_filename(),
        mime="application/json",
        width="stretch",
    )

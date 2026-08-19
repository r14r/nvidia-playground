from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd
import streamlit as st

from shared import APP_VERSION, model_ids, require_nvidia


def json_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


st.title("Run")
st.caption("Execute prompts and inspect NVIDIA NIM streaming chunks.")

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

    if not supports_streaming:
        st.session_state["streaming"] = False

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        key="temperature",
    )
    st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="top_p",
    )
    st.number_input(
        "Max Tokens",
        min_value=1,
        max_value=32768,
        step=128,
        key="max_tokens",
    )
    st.toggle(
        "Streaming",
        key="streaming",
        disabled=not supports_streaming,
    )
    st.toggle(
        "Raw chunk data",
        key="show_raw_chunks",
    )

    st.divider()
    st.caption("Selected model")
    st.code(selected_model, language=None)

can_run = nvidia.can_run(selected_model)

st.subheader("Model")
st.code(selected_model, language=None)

st.text_area(
    "System Prompt",
    key="system_prompt",
    height=100,
)

st.text_area(
    "Prompt",
    key="prompt",
    height=180,
)

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
    response_col, inspector_col = st.columns([2, 1])

    chunk_log: list[dict] = []
    raw_chunks: list[dict] = []
    full_response = ""
    started = time.perf_counter()
    first_content_ms = None

    try:
        if st.session_state["streaming"]:
            with response_col:
                st.subheader("Response")

                def render_stream() -> Iterator[str]:
                    first = None

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

                        if content and first is None:
                            first = float(event["elapsed_ms"])
                            st.session_state["_run_first_content_ms"] = first

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

                        if (
                            st.session_state["show_raw_chunks"]
                            and "raw" in event
                        ):
                            raw_chunks.append(event["raw"])

                        if content:
                            yield content

                st.session_state.pop("_run_first_content_ms", None)
                full_response = st.write_stream(render_stream(), cursor="▌")
                first_content_ms = st.session_state.pop(
                    "_run_first_content_ms",
                    None,
                )
        else:
            with response_col:
                st.subheader("Response")
                full_response = nvidia.query(
                    st.session_state["prompt"],
                    model=selected_model,
                    system_prompt=st.session_state["system_prompt"],
                    temperature=float(st.session_state["temperature"]),
                    top_p=float(st.session_state["top_p"]),
                    max_tokens=int(st.session_state["max_tokens"]),
                )
                st.markdown(full_response)

        total_time = time.perf_counter() - started

        st.divider()
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

        with inspector_col:
            st.subheader("Stream Inspector")

            if st.session_state["streaming"] and chunk_log:
                st.dataframe(
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
                    stat1, stat2 = st.columns(2)
                    stat1.metric(
                        "Avg. Gap",
                        f"{sum(gaps) / len(gaps):.1f} ms",
                    )
                    stat2.metric("Max Gap", f"{max(gaps):.1f} ms")

                if st.session_state["show_raw_chunks"]:
                    with st.expander("Raw chunks"):
                        st.json(raw_chunks)
            elif not st.session_state["streaming"]:
                st.info("Streaming is disabled.")
            else:
                st.info("No streaming chunks received.")

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
    st.subheader("Save Response")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        "Save response as JSON",
        data=json_bytes(st.session_state["last_response_payload"]),
        file_name=f"nvidia-response-{timestamp}.json",
        mime="application/json",
        width="stretch",
    )

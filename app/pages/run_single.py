from __future__ import annotations

import time
from datetime import datetime, timezone

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
    provider_model_info,
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

st.title("Single model")
st.caption(
    "Run one prompt against one model from the selected provider."
)

ensure_base_settings()

with st.sidebar:
    st.subheader("Run Settings")

    provider = st.selectbox(
        "Provider",
        options=PROVIDERS,
        key="run_provider",
        help=(
            "Ollama runs against localhost:11434. "
            "NVIDIA and models.json use the NVIDIA endpoints "
            "and credentials configured in models.json."
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
    st.warning(
        f"No runnable models are available from {provider}."
    )
    st.stop()

if st.session_state.get("selected_model") not in ids:
    st.session_state["selected_model"] = ids[0]

with st.sidebar:
    selected_model = st.selectbox(
        "Model",
        options=ids,
        key="selected_model",
    )

    supports_streaming = provider_supports_streaming(
        provider,
        selected_model,
        nvidia=nvidia,
    )
    render_runtime_settings(
        supports_streaming=supports_streaming
    )

can_run = provider_can_run(
    provider,
    selected_model,
    nvidia=nvidia,
)

render_prompt_tabs()

run = st.button(
    "Run Prompt",
    type="primary",
    width="stretch",
    disabled=not can_run,
)

if not can_run:
    if provider == "Ollama":
        st.warning(
            "The selected Ollama model is not available."
        )
    else:
        st.warning(
            "The selected NVIDIA model cannot run because its "
            "models.json entry does not contain a usable "
            "base_url and api_key."
        )

if run:
    console_run_prompt(
        selected_model,
        system_prompt=st.session_state["system_prompt"],
        user_prompt=st.session_state["prompt"],
    )
    console_connect(console_provider_name(provider))
    console_execute()
    console_waiting()

    response_tab, inspector_tab = st.tabs(
        ["Response", "Inspector"]
    )

    chunk_log: list[dict] = []
    raw_chunks: list[dict] = []
    full_response = ""
    started = time.perf_counter()
    first_content_ms = None
    timeout_seconds = float(
        st.session_state["timeout_seconds"]
    )

    with response_tab:
        response_placeholder = st.empty()
        response_metrics = st.empty()

    with inspector_tab:
        inspector_placeholder = st.empty()
        inspector_stats_placeholder = st.empty()
        raw_placeholder = st.empty()

    try:
        if st.session_state["streaming"]:
            for event in provider_stream_events(
                provider,
                st.session_state["prompt"],
                model=selected_model,
                nvidia=nvidia,
                system_prompt=st.session_state[
                    "system_prompt"
                ],
                temperature=float(
                    st.session_state["temperature"]
                ),
                top_p=float(st.session_state["top_p"]),
                max_tokens=int(
                    st.session_state["max_tokens"]
                ),
                include_raw=bool(
                    st.session_state["show_raw_chunks"]
                ),
                timeout_seconds=timeout_seconds,
            ):
                content = event.get("content", "")

                if (
                    content
                    and first_content_ms is None
                ):
                    first_content_ms = float(
                        event["elapsed_ms"]
                    )

                chunk_log.append(
                    {
                        "chunk": event["chunk"],
                        "time_ms": event["elapsed_ms"],
                        "gap_ms": event["gap_ms"],
                        "chars": event["chars"],
                        "delta": content,
                        "finish_reason": event.get(
                            "finish_reason"
                        ),
                    }
                )

                if (
                    st.session_state["show_raw_chunks"]
                    and "raw" in event
                ):
                    raw_chunks.append(event["raw"])

                if content:
                    full_response += content
                    response_placeholder.markdown(
                        full_response + "▌"
                    )

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
                        f"Avg. gap: "
                        f"{sum(gaps) / len(gaps):.1f} ms · "
                        f"Max gap: {max(gaps):.1f} ms"
                    )

            response_placeholder.markdown(full_response)
        else:
            full_response = provider_query(
                provider,
                st.session_state["prompt"],
                model=selected_model,
                nvidia=nvidia,
                system_prompt=st.session_state[
                    "system_prompt"
                ],
                temperature=float(
                    st.session_state["temperature"]
                ),
                top_p=float(st.session_state["top_p"]),
                max_tokens=int(
                    st.session_state["max_tokens"]
                ),
                timeout_seconds=timeout_seconds,
            )
            response_placeholder.markdown(full_response)
            inspector_placeholder.info(
                "Streaming is disabled."
            )

        total_time = time.perf_counter() - started

        with response_metrics.container():
            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )
            metric1.metric(
                "TTFT",
                (
                    f"{first_content_ms:.0f} ms"
                    if first_content_ms is not None
                    else "—"
                ),
            )
            metric2.metric(
                "Total Time",
                f"{total_time:.2f} s",
            )
            metric3.metric("Chunks", len(chunk_log))
            metric4.metric(
                "Characters",
                len(full_response),
            )

        if (
            st.session_state["streaming"]
            and not chunk_log
        ):
            inspector_placeholder.info(
                "No streaming chunks received."
            )

        if (
            st.session_state["show_raw_chunks"]
            and raw_chunks
        ):
            with raw_placeholder.container():
                with st.expander("Raw chunks"):
                    st.json(raw_chunks)

        export_info = provider_model_info(
            provider,
            selected_model,
            nvidia=nvidia,
        )
        export_info.pop("api_key", None)

        st.session_state["last_response_payload"] = {
            "schema_version": 1,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "application": {
                "name": "nvidia-playground",
                "version": APP_VERSION,
            },
            "provider": provider,
            "model": selected_model,
            "model_info": export_info,
            "request": {
                "system_prompt": st.session_state[
                    "system_prompt"
                ],
                "prompt": st.session_state["prompt"],
                "temperature": float(
                    st.session_state["temperature"]
                ),
                "top_p": float(
                    st.session_state["top_p"]
                ),
                "max_tokens": int(
                    st.session_state["max_tokens"]
                ),
                "timeout_seconds": timeout_seconds,
                "streaming": bool(
                    st.session_state["streaming"]
                ),
            },
            "response": {
                "text": full_response,
                "characters": len(full_response),
            },
            "metrics": {
                "ttft_ms": first_content_ms,
                "total_time_seconds": round(
                    total_time,
                    4,
                ),
                "chunks": len(chunk_log),
            },
            "chunks": chunk_log,
        }

        if st.session_state["show_raw_chunks"]:
            st.session_state[
                "last_response_payload"
            ]["raw_chunks"] = raw_chunks

        console_response(full_response)

    except TimeoutError as exc:
        console_error(
            selected_model,
            exc,
            partial_response=full_response,
        )
        st.error(str(exc))
        st.info(
            "Increase Timeout in the Run Settings sidebar "
            "and run the prompt again."
        )
    except Exception as exc:
        console_error(
            selected_model,
            exc,
            partial_response=full_response,
        )
        st.exception(exc)

if "last_response_payload" in st.session_state:
    st.divider()
    st.download_button(
        "Save response as JSON",
        data=json_bytes(
            st.session_state["last_response_payload"]
        ),
        file_name=response_filename(),
        mime="application/json",
        width="stretch",
    )

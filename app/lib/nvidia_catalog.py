from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

NVIDIA_MODELS_API_URL = "https://integrate.api.nvidia.com/v1/models"
NVIDIA_MODELS_CATALOG_URL = (
    "https://build.nvidia.com/models"
    "?filters=nimType%3Anim_type_preview&pageSize=200"
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 nvidia-playground/0.8.4",
    "Accept": "application/json",
}


def _request_json(
    url: str,
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    request = Request(url, headers=_HEADERS)
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("NVIDIA models endpoint returned invalid JSON.")
    return payload


def parse_nvidia_models_payload(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            "NVIDIA models endpoint did not return a model list."
        )

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            continue

        model_id = str(item.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)

        owner = str(
            item.get("owned_by")
            or model_id.split("/", 1)[0]
            or ""
        )
        rows.append(
            {
                "Provider": "NVIDIA",
                "Model": model_id,
                "Owner": owner,
                "Object": item.get("object") or "",
                "Created": item.get("created") or "",
                "Max Model Length": item.get("max_model_len") or "",
                "URL": f"https://build.nvidia.com/{model_id}",
                "Source": NVIDIA_MODELS_API_URL,
            }
        )

    if not rows:
        raise RuntimeError(
            "NVIDIA models endpoint was reachable, but returned no models."
        )

    return sorted(rows, key=lambda row: str(row["Model"]).lower())


def list_nvidia_api_models(
    *,
    timeout_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    return parse_nvidia_models_payload(
        _request_json(
            NVIDIA_MODELS_API_URL,
            timeout_seconds=timeout_seconds,
        )
    )

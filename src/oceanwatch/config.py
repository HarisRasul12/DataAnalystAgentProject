from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str
    vertex_model: str
    vertex_model_candidates: tuple[str, ...]
    gcp_project: str | None
    gcp_region: str
    adk_enabled: bool
    require_adk_success: bool
    allow_transient_fallback_when_strict: bool
    default_hours_back: int
    noaa_timeout_seconds: float
    noaa_retry_attempts: int
    noaa_retry_wait_seconds: float
    adk_timeout_seconds: float


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_model_candidates(primary_model: str, value: str | None) -> tuple[str, ...]:
    candidates = [primary_model]
    if value:
        for item in value.split(","):
            model = item.strip()
            if model:
                candidates.append(model)

    seen: set[str] = set()
    unique: list[str] = []
    for model in candidates:
        if model not in seen:
            seen.add(model)
            unique.append(model)
    return tuple(unique)


def load_settings() -> Settings:
    gcp_project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or None
    )

    return Settings(
        app_name=os.getenv("OCEANWATCH_APP_NAME", "oceanwatch"),
        vertex_model=os.getenv("VERTEX_MODEL", "gemini-2.0-flash-lite"),
        vertex_model_candidates=_as_model_candidates(
            os.getenv("VERTEX_MODEL", "gemini-2.0-flash-lite"),
            os.getenv("OCEANWATCH_VERTEX_MODEL_CANDIDATES", "gemini-2.0-flash,gemini-2.5-flash-lite"),
        ),
        gcp_project=gcp_project,
        gcp_region=os.getenv("GOOGLE_CLOUD_LOCATION", os.getenv("OCEANWATCH_REGION", "us-central1")),
        adk_enabled=_as_bool(os.getenv("OCEANWATCH_ADK_ENABLED", "true"), default=True),
        require_adk_success=_as_bool(os.getenv("OCEANWATCH_REQUIRE_ADK_SUCCESS", "true"), default=True),
        allow_transient_fallback_when_strict=_as_bool(
            os.getenv("OCEANWATCH_ALLOW_TRANSIENT_FALLBACK_WHEN_STRICT", "true"),
            default=True,
        ),
        default_hours_back=int(os.getenv("OCEANWATCH_DEFAULT_HOURS", "168")),
        noaa_timeout_seconds=float(os.getenv("OCEANWATCH_TIMEOUT_SECONDS", "20")),
        noaa_retry_attempts=int(os.getenv("OCEANWATCH_RETRY_ATTEMPTS", "3")),
        noaa_retry_wait_seconds=float(os.getenv("OCEANWATCH_RETRY_WAIT_SECONDS", "1.5")),
        adk_timeout_seconds=float(os.getenv("OCEANWATCH_ADK_TIMEOUT_SECONDS", "40")),
    )

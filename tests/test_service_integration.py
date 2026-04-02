from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from oceanwatch.config import Settings
from oceanwatch.schemas import AnalysisRequest, SourcePayload
from oceanwatch.service import OceanWatchService
import oceanwatch.service as service_module


class FakeNOAAClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch_ndbc_observations(self, station_id: str, hours_back: int):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-29", periods=6, freq="h", tz="UTC"),
                "wave_height_m": [0.4, 0.5, 0.6, 0.55, 0.6, 0.65],
                "wind_speed_mps": [3.5, 4.0, 4.2, 4.1, 4.3, 4.4],
                "gust_mps": [5.0, 5.1, 5.3, 5.2, 5.4, 5.5],
                "water_temp_c": [13.0, 13.1, 13.1, 13.2, 13.2, 13.3],
            }
        )
        payload = SourcePayload(
            source_name="ndbc_buoy",
            station_id=station_id,
            available=True,
            row_count=len(frame),
            records=[],
            metadata={"provider": "fake"},
            fetched_at_utc=datetime.now(timezone.utc),
        )
        return frame, payload

    def fetch_coops_water_level(self, station_id: str, begin, end):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-29", periods=6, freq="h", tz="UTC"),
                "water_level_m": [0.1, 0.15, 0.2, 0.18, 0.17, 0.16],
                "sigma": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                "quality_flag": ["p"] * 6,
            }
        )
        payload = SourcePayload(
            source_name="coops_water_level",
            station_id=station_id,
            available=True,
            row_count=len(frame),
            records=[],
            metadata={"provider": "fake"},
            fetched_at_utc=datetime.now(timezone.utc),
        )
        return frame, payload

    def fetch_coops_tide_predictions(self, station_id: str, begin, end):
        raise RuntimeError("simulated tide endpoint outage")


def test_service_graceful_degradation_when_one_source_fails() -> None:
    settings = Settings(
        app_name="oceanwatch-test",
        vertex_model="gemini-2.0-flash-lite",
        vertex_model_candidates=("gemini-2.0-flash-lite",),
        gcp_project=None,
        gcp_region="us-central1",
        adk_enabled=False,
        require_adk_success=False,
        allow_transient_fallback_when_strict=True,
        default_hours_back=168,
        noaa_timeout_seconds=20,
        noaa_retry_attempts=1,
        noaa_retry_wait_seconds=0.1,
        adk_timeout_seconds=15,
    )

    service = OceanWatchService(settings=settings, client=FakeNOAAClient(settings))
    request = AnalysisRequest(
        user_question="Give me the main coastal signal and risks.",
        station_key="sf_bay",
        hours_back=72,
    )

    result = service.run_analysis(request)

    assert result.health["ndbc_buoy"] == "ok"
    assert result.health["coops_water_level"] == "ok"
    assert result.health["coops_tide_predictions"] == "error"
    assert len(result.insight.thesis) > 10
    assert result.metrics.metric_cards["buoy_rows"] == 6


def test_strict_mode_returns_fallback_thesis_when_adk_unavailable() -> None:
    settings = Settings(
        app_name="oceanwatch-test",
        vertex_model="gemini-2.0-flash-lite",
        vertex_model_candidates=("gemini-2.0-flash-lite",),
        gcp_project=None,
        gcp_region="us-central1",
        adk_enabled=False,
        require_adk_success=True,
        allow_transient_fallback_when_strict=True,
        default_hours_back=168,
        noaa_timeout_seconds=20,
        noaa_retry_attempts=1,
        noaa_retry_wait_seconds=0.1,
        adk_timeout_seconds=15,
    )

    service = OceanWatchService(settings=settings, client=FakeNOAAClient(settings))
    request = AnalysisRequest(
        user_question="Generate a graded hypothesis.",
        station_key="sf_bay",
        hours_back=72,
    )

    result = service.run_analysis(request)

    assert result.adk_status == "failed_optional"
    assert result.adk_attempted is False
    assert len(result.insight.thesis) > 10
    assert result.adk_error_summary is not None


def test_adk_success_path_returns_adk_thesis(monkeypatch) -> None:
    settings = Settings(
        app_name="oceanwatch-test",
        vertex_model="gemini-2.0-flash-lite",
        vertex_model_candidates=("gemini-2.0-flash-lite",),
        gcp_project=None,
        gcp_region="us-central1",
        adk_enabled=True,
        require_adk_success=True,
        allow_transient_fallback_when_strict=True,
        default_hours_back=168,
        noaa_timeout_seconds=20,
        noaa_retry_attempts=1,
        noaa_retry_wait_seconds=0.1,
        adk_timeout_seconds=15,
    )

    monkeypatch.setattr(service_module, "adk_is_available", lambda: True)

    def _fake_adk_run(self, runtime):
        runtime.adk_status = "success"
        runtime.adk_model_used = "gemini-2.0-flash-lite"
        runtime.state_delta = {
            "insight_report": {
                "thesis": "ADK generated thesis.",
                "evidence_bullets": ["Evidence from ADK path."],
                "notable_anomalies": [],
                "limitations": [],
                "recommended_followups": [],
            }
        }

    monkeypatch.setattr(OceanWatchService, "_run_adk_with_model_candidates", _fake_adk_run)

    service = OceanWatchService(settings=settings, client=FakeNOAAClient(settings))
    result = service.run_analysis(
        AnalysisRequest(
            user_question="Give me ADK synthesis.",
            station_key="sf_bay",
            hours_back=48,
        )
    )

    assert result.adk_attempted is True
    assert result.adk_status == "success"
    assert result.insight.thesis == "ADK generated thesis."


def test_transient_adk_failure_returns_labeled_fallback(monkeypatch) -> None:
    settings = Settings(
        app_name="oceanwatch-test",
        vertex_model="gemini-2.0-flash-lite",
        vertex_model_candidates=("gemini-2.0-flash-lite",),
        gcp_project=None,
        gcp_region="us-central1",
        adk_enabled=True,
        require_adk_success=True,
        allow_transient_fallback_when_strict=True,
        default_hours_back=168,
        noaa_timeout_seconds=20,
        noaa_retry_attempts=1,
        noaa_retry_wait_seconds=0.1,
        adk_timeout_seconds=15,
    )

    monkeypatch.setattr(service_module, "adk_is_available", lambda: True)

    def _fake_adk_fail(self, runtime):
        runtime.adk_status = "failed_optional"
        runtime.adk_error_category = "network"
        runtime.adk_error_summary = "Simulated network timeout."

    monkeypatch.setattr(OceanWatchService, "_run_adk_with_model_candidates", _fake_adk_fail)

    service = OceanWatchService(settings=settings, client=FakeNOAAClient(settings))
    result = service.run_analysis(
        AnalysisRequest(
            user_question="Give me robust thesis.",
            station_key="sf_bay",
            hours_back=48,
        )
    )

    assert result.adk_attempted is True
    assert result.adk_status == "failed_transient_fallback"
    assert len(result.insight.thesis) > 10
    assert any(step.get("step") == "run.insight_transient_fallback" for step in result.execution_trace)

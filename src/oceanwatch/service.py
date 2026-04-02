from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from uuid import uuid4

import pandas as pd

from oceanwatch.agents import adk_is_available, create_oceanwatch_root_agent, create_wave_monte_carlo_agent
from oceanwatch.config import Settings, load_settings
from oceanwatch.metrics import compute_ocean_metrics, compute_wave_monte_carlo, metric_lookup
from oceanwatch.noaa_clients import NOAAClient, default_begin_end
from oceanwatch.schemas import AnalysisRequest, InsightReport, RunResult, SourcePayload, WaveMonteCarloReport
from oceanwatch.schemas import PostMonteCarloHypothesis, WaveHypothesisInterpretation, WaveSimulationSnapshot
from oceanwatch.stations import StationCatalogEntry, get_station_by_key
from oceanwatch.visuals import build_visuals as build_visual_figures
from oceanwatch.visuals import visual_summary


@dataclass
class ToolRuntime:
    request: AnalysisRequest
    station: StationCatalogEntry
    buoy_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    water_level_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    tide_prediction_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    payloads: dict[str, SourcePayload] = field(default_factory=dict)
    figures: dict[str, object] = field(default_factory=dict)
    health: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    state_delta: dict[str, object] = field(default_factory=dict)
    adk_status: str = "not_attempted"
    adk_attempted: bool = False
    adk_error_summary: str | None = None
    adk_error_category: str | None = None
    adk_model_used: str | None = None
    cached_metrics_report: object | None = None
    execution_trace: list[dict[str, object]] = field(default_factory=list)


class OceanWatchToolset:
    def __init__(self, settings: Settings, client: NOAAClient, runtime: ToolRuntime):
        self.settings = settings
        self.client = client
        self.runtime = runtime

    def tools(self):
        return [
            self.get_ndbc_observations,
            self.get_coops_water_level,
            self.get_coops_tide_predictions,
            self.compute_ocean_metrics,
            self.compute_risk_analytics,
            self.simulate_coastal_scenarios,
            self.build_visuals,
        ]

    def get_ndbc_observations(self, station_id: str | None = None, hours_back: int | None = None) -> dict:
        if "ndbc_buoy" in self.runtime.payloads:
            _trace(self.runtime, "tool.get_ndbc_observations", details="cache hit")
            return _json_safe(_tool_payload_brief(self.runtime.payloads["ndbc_buoy"]))

        sid = station_id or self.runtime.station.ndbc_station_id
        horizon = int(hours_back or self.runtime.request.hours_back)
        started = time.perf_counter()
        try:
            frame, payload = self.client.fetch_ndbc_observations(sid, horizon)
            self.runtime.buoy_frame = frame
            self.runtime.cached_metrics_report = None
            self.runtime.payloads["ndbc_buoy"] = payload
            self.runtime.health["ndbc_buoy"] = "ok" if payload.available else "empty"
            _trace(
                self.runtime,
                "tool.get_ndbc_observations",
                details=f"rows={len(frame)}, station={sid}",
                duration_ms=_elapsed_ms(started),
            )
            return _json_safe(_tool_payload_brief(payload))
        except Exception as exc:
            payload = _unavailable_payload("ndbc_buoy", sid, str(exc))
            self.runtime.payloads["ndbc_buoy"] = payload
            self.runtime.health["ndbc_buoy"] = "error"
            self.runtime.warnings.append(f"NDBC fetch failed: {exc}")
            _trace(
                self.runtime,
                "tool.get_ndbc_observations",
                status="error",
                details=str(exc),
                duration_ms=_elapsed_ms(started),
            )
            return _json_safe(_tool_payload_brief(payload))

    def get_coops_water_level(self, station_id: str | None = None) -> dict:
        if "coops_water_level" in self.runtime.payloads:
            _trace(self.runtime, "tool.get_coops_water_level", details="cache hit")
            return _json_safe(_tool_payload_brief(self.runtime.payloads["coops_water_level"]))

        sid = station_id or self.runtime.station.coops_station_id
        begin, end = default_begin_end(self.runtime.request.hours_back)
        started = time.perf_counter()
        try:
            frame, payload = self.client.fetch_coops_water_level(sid, begin, end)
            self.runtime.water_level_frame = frame
            self.runtime.cached_metrics_report = None
            self.runtime.payloads["coops_water_level"] = payload
            self.runtime.health["coops_water_level"] = "ok" if payload.available else "empty"
            _trace(
                self.runtime,
                "tool.get_coops_water_level",
                details=f"rows={len(frame)}, station={sid}",
                duration_ms=_elapsed_ms(started),
            )
            return _json_safe(_tool_payload_brief(payload))
        except Exception as exc:
            payload = _unavailable_payload("coops_water_level", sid, str(exc))
            self.runtime.payloads["coops_water_level"] = payload
            self.runtime.health["coops_water_level"] = "error"
            self.runtime.warnings.append(f"CO-OPS water level fetch failed: {exc}")
            _trace(
                self.runtime,
                "tool.get_coops_water_level",
                status="error",
                details=str(exc),
                duration_ms=_elapsed_ms(started),
            )
            return _json_safe(_tool_payload_brief(payload))

    def get_coops_tide_predictions(self, station_id: str | None = None) -> dict:
        if "coops_tide_predictions" in self.runtime.payloads:
            _trace(self.runtime, "tool.get_coops_tide_predictions", details="cache hit")
            return _json_safe(_tool_payload_brief(self.runtime.payloads["coops_tide_predictions"]))

        sid = station_id or self.runtime.station.coops_station_id
        begin, end = default_begin_end(self.runtime.request.hours_back)
        started = time.perf_counter()
        try:
            frame, payload = self.client.fetch_coops_tide_predictions(sid, begin, end)
            self.runtime.tide_prediction_frame = frame
            self.runtime.cached_metrics_report = None
            self.runtime.payloads["coops_tide_predictions"] = payload
            self.runtime.health["coops_tide_predictions"] = "ok" if payload.available else "empty"
            _trace(
                self.runtime,
                "tool.get_coops_tide_predictions",
                details=f"rows={len(frame)}, station={sid}",
                duration_ms=_elapsed_ms(started),
            )
            return _json_safe(_tool_payload_brief(payload))
        except Exception as exc:
            payload = _unavailable_payload("coops_tide_predictions", sid, str(exc))
            self.runtime.payloads["coops_tide_predictions"] = payload
            self.runtime.health["coops_tide_predictions"] = "error"
            self.runtime.warnings.append(f"CO-OPS predictions fetch failed: {exc}")
            _trace(
                self.runtime,
                "tool.get_coops_tide_predictions",
                status="error",
                details=str(exc),
                duration_ms=_elapsed_ms(started),
            )
            return _json_safe(_tool_payload_brief(payload))

    def compute_ocean_metrics(self, source: str = "all") -> dict:
        self._ensure_data_collected()
        started = time.perf_counter()
        report = self._get_metrics_report()
        _trace(
            self.runtime,
            "tool.compute_ocean_metrics",
            details=f"source={source}",
            duration_ms=_elapsed_ms(started),
        )
        payload = report.model_dump()
        if source == "buoy":
            payload = {
                "metric_cards": payload["metric_cards"],
                "buoy_metrics": payload["buoy_metrics"],
                "confidence_note": payload["confidence_note"],
            }
        elif source == "coops":
            payload = {
                "metric_cards": payload["metric_cards"],
                "water_level_metrics": payload["water_level_metrics"],
                "tide_metrics": payload["tide_metrics"],
                "confidence_note": payload["confidence_note"],
            }
        return _json_safe(payload)

    def compute_risk_analytics(self) -> dict:
        self._ensure_data_collected()
        started = time.perf_counter()
        report = self._get_metrics_report()
        _trace(
            self.runtime,
            "tool.compute_risk_analytics",
            details="severity/regression/distribution/irregular-patterns",
            duration_ms=_elapsed_ms(started),
        )
        return _json_safe(
            {
                "severity": report.advanced_analytics.get("severity", {}),
                "regression_signals": report.advanced_analytics.get("regression_signals", []),
                "distribution_profiles": report.advanced_analytics.get("distribution_profiles", []),
                "irregular_patterns": report.advanced_analytics.get("irregular_patterns", []),
                "risk_headline": report.advanced_analytics.get("risk_headline", ""),
            }
        )

    def simulate_coastal_scenarios(self) -> dict:
        self._ensure_data_collected()
        started = time.perf_counter()
        report = self._get_metrics_report()
        outcomes = report.advanced_analytics.get("simulation_outcomes", [])
        _trace(
            self.runtime,
            "tool.simulate_coastal_scenarios",
            details=f"scenario_count={len(outcomes)}",
            duration_ms=_elapsed_ms(started),
        )
        return _json_safe({"simulation_outcomes": outcomes, "count": len(outcomes)})

    def build_visuals(self, source: str = "overview") -> dict:
        self._ensure_data_collected()
        started = time.perf_counter()
        report = self._get_metrics_report()
        self.runtime.figures = build_visual_figures(
            self.runtime.buoy_frame,
            self.runtime.water_level_frame,
            self.runtime.tide_prediction_frame,
            report.advanced_analytics,
        )
        summary = visual_summary(self.runtime.figures)
        _trace(
            self.runtime,
            "tool.build_visuals",
            details=f"figure_count={len(self.runtime.figures)}",
            duration_ms=_elapsed_ms(started),
        )
        if source == "risk":
            return _json_safe({"risk_distribution": summary.get("risk_distribution", {})})
        return _json_safe(summary)

    def _ensure_data_collected(self) -> None:
        if "ndbc_buoy" not in self.runtime.payloads:
            self.get_ndbc_observations()
        if "coops_water_level" not in self.runtime.payloads:
            self.get_coops_water_level()
        if "coops_tide_predictions" not in self.runtime.payloads:
            self.get_coops_tide_predictions()

    def _get_metrics_report(self):
        if self.runtime.cached_metrics_report is None:
            self.runtime.cached_metrics_report = compute_ocean_metrics(
                self.runtime.buoy_frame,
                self.runtime.water_level_frame,
                self.runtime.tide_prediction_frame,
            )
        return self.runtime.cached_metrics_report


class OceanWatchService:
    def __init__(self, settings: Settings | None = None, client: NOAAClient | None = None):
        self.settings = settings or load_settings()
        self.client = client or NOAAClient(self.settings)
        self._last_successful_model: str | None = None

    def run_analysis(self, request: AnalysisRequest) -> RunResult:
        run_started = time.perf_counter()
        station = get_station_by_key(request.station_key)
        runtime = ToolRuntime(request=request, station=station)
        adk_ready = self.settings.adk_enabled and adk_is_available()
        _trace(
            runtime,
            "run.request_received",
            details=(
                f"station={station.key}, hours_back={request.hours_back}, "
                f"adk_enabled={self.settings.adk_enabled}, adk_ready={adk_ready}"
            ),
        )

        # Pre-fetch NOAA sources once. ADK agents still call tools, but those calls now
        # reuse cached runtime payloads to reduce token and latency pressure.
        prefetch_started = time.perf_counter()
        self._prefetch_sources(runtime)
        _trace(runtime, "run.prefetch_complete", duration_ms=_elapsed_ms(prefetch_started))

        if self.settings.require_adk_success and not adk_ready:
            runtime.adk_status = "failed_optional"
            runtime.adk_error_category = "config"
            runtime.adk_error_summary = (
                "ADK runtime is required but unavailable. "
                "Install google-adk/google-genai and verify Vertex credentials."
            )
            runtime.warnings.append(runtime.adk_error_summary)
            _trace(runtime, "run.adk_unavailable", status="error", details=runtime.adk_error_summary)
        elif adk_ready:
            runtime.adk_attempted = True
            adk_started = time.perf_counter()
            self._run_adk_with_model_candidates(runtime)
            _trace(
                runtime,
                "run.adk_phase_complete",
                details=f"status={runtime.adk_status}, model={runtime.adk_model_used}",
                duration_ms=_elapsed_ms(adk_started),
            )
        else:
            runtime.adk_status = "skipped"
            runtime.warnings.append("ADK disabled: running data collection and deterministic analytics only.")
            _trace(runtime, "run.adk_skipped", details="OCEANWATCH_ADK_ENABLED=false")

        if not runtime.payloads:
            self._collect_direct(runtime)

        metrics_report = compute_ocean_metrics(
            runtime.buoy_frame,
            runtime.water_level_frame,
            runtime.tide_prediction_frame,
        )
        _trace(runtime, "run.metrics_computed", details="deterministic EDA complete")

        if not runtime.figures:
            runtime.figures = build_visual_figures(
                runtime.buoy_frame,
                runtime.water_level_frame,
                runtime.tide_prediction_frame,
                metrics_report.advanced_analytics,
            )
            _trace(runtime, "run.visuals_computed", details=f"figure_count={len(runtime.figures)}")

        insight = self._insight_from_state(runtime.state_delta)
        if insight is None and runtime.adk_status in {"success", "recovery_success"}:
            runtime.adk_status = "failed_optional"
            runtime.adk_error_category = runtime.adk_error_category or "schema"
            runtime.adk_error_summary = runtime.adk_error_summary or (
                "ADK run completed but did not return a structured InsightReport."
            )
            runtime.warnings.append(runtime.adk_error_summary)
            _trace(runtime, "run.adk_schema_missing", status="error", details=runtime.adk_error_summary)

        if insight is None:
            if (
                runtime.adk_error_category in {"network", "rate_limit"}
                and self.settings.allow_transient_fallback_when_strict
            ):
                insight = self._fallback_insight(runtime, metrics_report)
                runtime.adk_status = "failed_transient_fallback"
                runtime.warnings.append(
                    "ADK transient failure detected (network/rate-limit); deterministic hypothesis fallback returned."
                )
                _trace(runtime, "run.insight_transient_fallback", details="strict mode bypassed for transient ADK failure")
            else:
                insight = self._fallback_insight(runtime, metrics_report)
                runtime.adk_status = "failed_optional" if runtime.adk_attempted else runtime.adk_status
                _trace(runtime, "run.insight_fallback", details="deterministic fallback generated")
        else:
            _trace(runtime, "run.insight_generated", details="InsightReport available")

        payloads = {
            "ndbc_buoy": runtime.payloads.get(
                "ndbc_buoy",
                _unavailable_payload("ndbc_buoy", station.ndbc_station_id, "No request executed."),
            ),
            "coops_water_level": runtime.payloads.get(
                "coops_water_level",
                _unavailable_payload("coops_water_level", station.coops_station_id, "No request executed."),
            ),
            "coops_tide_predictions": runtime.payloads.get(
                "coops_tide_predictions",
                _unavailable_payload("coops_tide_predictions", station.coops_station_id, "No request executed."),
            ),
        }

        return RunResult(
            request=request,
            station=asdict(station),
            health={
                "ndbc_buoy": runtime.health.get("ndbc_buoy", "missing"),
                "coops_water_level": runtime.health.get("coops_water_level", "missing"),
                "coops_tide_predictions": runtime.health.get("coops_tide_predictions", "missing"),
            },
            source_payloads=payloads,
            metrics=metrics_report,
            insight=insight,
            figures=runtime.figures,
            tables={
                "buoy": runtime.buoy_frame,
                "water_level": runtime.water_level_frame,
                "tide_predictions": runtime.tide_prediction_frame,
            },
            warnings=_dedupe(runtime.warnings),
            adk_status=runtime.adk_status,
            adk_attempted=runtime.adk_attempted,
            adk_error_summary=runtime.adk_error_summary,
            adk_model_used=runtime.adk_model_used,
            execution_trace=runtime.execution_trace,
            runtime_seconds=round(time.perf_counter() - run_started, 3),
        )

    def run_wave_monte_carlo_agent(
        self,
        result: RunResult,
        days_ahead: int = 3,
        path_count: int = 800,
    ) -> WaveMonteCarloReport:
        buoy_frame = result.tables.get("buoy", pd.DataFrame())
        station_name = str(result.station.get("display_name", "Selected station"))
        horizon_days = int(max(1, min(7, days_ahead)))
        paths = int(max(200, min(4000, path_count)))

        deterministic_payload = compute_wave_monte_carlo(
            buoy_frame=buoy_frame,
            days_ahead=horizon_days,
            path_count=paths,
        )
        deterministic_report = self._deterministic_wave_monte_carlo_report(
            deterministic_payload=deterministic_payload,
            station_name=station_name,
            source="deterministic_tool",
        )

        if not (self.settings.adk_enabled and adk_is_available()):
            deterministic_report.status = "adk_unavailable"
            deterministic_report.mc_adk_status = "blocked_required"
            deterministic_report.mc_hypothesis_ready = False
            deterministic_report.mc_adk_attempts = ["adk_unavailable"]
            deterministic_report.mc_adk_hypothesis = None
            deterministic_report.post_mc_hypothesis = None
            deterministic_report.plain_english_summary = ""
            deterministic_report.limitations.insert(
                0,
                "Monte Carlo Hypothesis blocked: ADK interpretation required. "
                "ADK runtime is unavailable for post-simulation reasoning.",
            )
            return deterministic_report

        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.settings.gcp_region)
        if self.settings.gcp_project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.settings.gcp_project)

        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        snapshot = WaveSimulationSnapshot(
            station_name=station_name,
            horizon_hours=deterministic_report.horizon_hours,
            path_count=deterministic_report.path_count,
            model_name=deterministic_report.model_name,
            final_p50_m=deterministic_report.final_p50_m,
            final_p90_m=deterministic_report.final_p90_m,
            expected_peak_m=deterministic_report.expected_peak_m,
            cvar95_final_m=deterministic_report.cvar95_final_m,
            probability_exceed_2m=deterministic_report.probability_exceed_2m,
            probability_exceed_3m=deterministic_report.probability_exceed_3m,
            probability_reach_3m_anytime=deterministic_report.probability_reach_3m_anytime,
            mean_reversion_kappa=deterministic_report.mean_reversion_kappa,
            volatility_per_sqrt_hour=deterministic_report.volatility_per_sqrt_hour,
            jump_intensity_per_hour=deterministic_report.jump_intensity_per_hour,
            limitations=deterministic_report.limitations[:],
        )

        def load_simulation_snapshot() -> dict:
            return _json_safe(snapshot.model_dump())
        advanced = result.metrics.advanced_analytics or {}
        recent_wave_records: list[dict[str, object]] = []
        if not buoy_frame.empty and {"timestamp", "wave_height_m"}.issubset(set(buoy_frame.columns)):
            clipped = buoy_frame[["timestamp", "wave_height_m"]].dropna().tail(48).copy()
            recent_wave_records = [
                {
                    "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                    "wave_height_m": round(float(row["wave_height_m"]), 3),
                }
                for _, row in clipped.iterrows()
            ]

        wave_context = _json_safe(
            {
                "station": result.station,
                "question": result.request.user_question,
                "wave_outlook": advanced.get("wave_outlook", {}),
                "severity": advanced.get("severity", {}),
                "irregular_patterns": advanced.get("irregular_patterns", [])[:4],
                "distribution_profiles": advanced.get("distribution_profiles", []),
                "recent_wave_records": recent_wave_records,
                "preliminary_hypothesis": result.insight.thesis,
                "preliminary_narrative": result.insight.narrative_paragraphs[:2],
            }
        )

        prompt = (
            "Run the wave Monte Carlo specialist analysis. "
            f"Station: {station_name}. Horizon days: {horizon_days}. Paths: {paths}. "
            "The deterministic simulation is already computed; perform ADK interpretation and post-hypothesis reconciliation."
        )

        def _run_attempt(model_name: str, timeout_seconds: float) -> tuple[WaveMonteCarloReport, bool]:
            agent = create_wave_monte_carlo_agent(
                self.settings,
                tools=[load_simulation_snapshot],
                model_name=model_name,
            )
            session_service = InMemorySessionService()
            runner = Runner(agent=agent, app_name=self.settings.app_name, session_service=session_service)
            session_id = f"oceanwatch-wave-mc-{uuid4().hex[:8]}"

            async def _drive() -> dict[str, object]:
                await session_service.create_session(
                    app_name=self.settings.app_name,
                    user_id=result.request.user_id,
                    session_id=session_id,
                    state={
                        "station_key": result.request.station_key,
                        "preliminary_hypothesis": result.insight.thesis,
                        "preliminary_narrative": result.insight.narrative_paragraphs[:2],
                        "wave_context": wave_context,
                        "simulation_snapshot": snapshot.model_dump(),
                    },
                )
                message = Content(role="user", parts=[Part(text=prompt)])
                captured_state: dict[str, object] = {}
                async for event in runner.run_async(
                    user_id=result.request.user_id,
                    session_id=session_id,
                    new_message=message,
                ):
                    actions = getattr(event, "actions", None)
                    delta = getattr(actions, "state_delta", None) if actions else None
                    if isinstance(delta, dict):
                        captured_state.update(delta)

                session = await session_service.get_session(
                    app_name=self.settings.app_name,
                    user_id=result.request.user_id,
                    session_id=session_id,
                )
                if hasattr(session, "state") and isinstance(session.state, dict):
                    captured_state.update(session.state)
                return captured_state

            state = asyncio.run(asyncio.wait_for(_drive(), timeout=timeout_seconds))
            interpretation = self._wave_hypothesis_interpretation_from_state(state)
            post_report = self._post_monte_carlo_hypothesis_from_state(state)
            if interpretation is None:
                available_keys = ", ".join(sorted(state.keys())[:25]) if state else "none"
                raise RuntimeError(
                    "ADK Monte Carlo response missing WaveHypothesisInterpretation output. "
                    f"Available state keys: {available_keys}"
                )
            if post_report is None:
                available_keys = ", ".join(sorted(state.keys())[:25]) if state else "none"
                raise RuntimeError(
                    "ADK Monte Carlo response missing PostMonteCarloHypothesis output. "
                    f"Available state keys: {available_keys}"
                )

            report = deterministic_report.model_copy(deep=True)
            report.source = "deterministic_engine_plus_adk"
            report.status = "ok"
            report.mc_hypothesis_ready = True
            report.mc_adk_hypothesis = interpretation.mc_adk_hypothesis
            report.plain_english_summary = interpretation.plain_english_summary
            report.post_mc_hypothesis = post_report.post_mc_hypothesis
            report.interpretation_points = interpretation.interpretation_points + [
                f"Post-Monte-Carlo shift: {item}" for item in post_report.what_changed
            ]
            merged_limitations = report.limitations + interpretation.limitations
            report.limitations = _dedupe(merged_limitations)
            return report, True

        primary_model = self._last_successful_model or self.settings.vertex_model
        candidate_models = [primary_model]
        for candidate in self.settings.vertex_model_candidates:
            if candidate not in candidate_models:
                candidate_models.append(candidate)

        timeout_seconds = max(15.0, min(30.0, self.settings.adk_timeout_seconds * 0.75))
        last_exc: Exception | None = None
        last_category: str = "unknown"
        attempts: list[str] = []
        retried_success = False

        for idx, model_name in enumerate(candidate_models):
            try:
                report, _ = _run_attempt(model_name, timeout_seconds=timeout_seconds)
                self._last_successful_model = model_name
                report.mc_adk_status = "retry_success" if retried_success else "success"
                report.mc_adk_attempts = attempts or [f"{model_name}:success"]
                return report
            except Exception as exc:
                last_exc = exc
                details = _flatten_exception_messages(exc)
                category, _ = _classify_adk_error(details)
                last_category = category
                attempts.append(f"{model_name}:{category}")

                if category in {"network", "rate_limit"}:
                    try:
                        report, _ = _run_attempt(model_name, timeout_seconds=max(12.0, timeout_seconds - 5.0))
                        self._last_successful_model = model_name
                        retried_success = True
                        report.mc_adk_status = "retry_success"
                        report.mc_adk_attempts = attempts + [f"{model_name}:retry-success"]
                        return report
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        retry_category, _ = _classify_adk_error(_flatten_exception_messages(retry_exc))
                        last_category = retry_category
                        attempts.append(f"{model_name}:retry-{retry_category}")

                if category not in {"model_config", "unknown", "schema"}:
                    break
                if idx >= 1:
                    break

        for model_name in candidate_models[:2]:
            try:
                recovery_insight = self._run_post_mc_hypothesis_recovery(
                    result=result,
                    deterministic_report=deterministic_report,
                    model_name=model_name,
                    timeout_seconds=max(16.0, min(28.0, timeout_seconds)),
                )
                if recovery_insight is None:
                    continue
                self._last_successful_model = model_name
                report = deterministic_report.model_copy(deep=True)
                report.source = "adk_hypothesis_recovery_agent"
                report.status = "ok"
                report.mc_adk_status = "retry_success"
                report.mc_hypothesis_ready = True
                report.mc_adk_attempts = attempts + [f"{model_name}:hypothesis-recovery-success"]
                report.mc_adk_hypothesis = recovery_insight.thesis
                report.plain_english_summary = (
                    recovery_insight.narrative_paragraphs[0]
                    if recovery_insight.narrative_paragraphs
                    else recovery_insight.thesis
                )
                report.post_mc_hypothesis = (
                    recovery_insight.narrative_paragraphs[1]
                    if len(recovery_insight.narrative_paragraphs) > 1
                    else recovery_insight.thesis
                )
                report.interpretation_points = _dedupe(
                    report.interpretation_points + recovery_insight.evidence_bullets[:4]
                )
                report.limitations = _dedupe(report.limitations + recovery_insight.limitations)
                return report
            except Exception as recovery_exc:
                details = _flatten_exception_messages(recovery_exc)
                category, _ = _classify_adk_error(details)
                attempts.append(f"{model_name}:hypothesis-recovery-{category}")
                last_exc = recovery_exc
                last_category = category

        deterministic_report.status = "adk_blocked"
        deterministic_report.mc_adk_status = "blocked_required"
        deterministic_report.mc_hypothesis_ready = False
        deterministic_report.mc_adk_hypothesis = None
        deterministic_report.post_mc_hypothesis = None
        deterministic_report.plain_english_summary = ""
        category, action = _classify_adk_error(_flatten_exception_messages(last_exc))
        attempt_text = "; ".join(attempts) if attempts else f"{primary_model}:{last_category}"
        deterministic_report.mc_adk_attempts = attempts if attempts else [f"{primary_model}:{last_category}"]
        deterministic_report.limitations.insert(
            0,
            "Monte Carlo Hypothesis blocked: ADK interpretation required. "
            f"ADK Monte Carlo agent failed ({category}). {action} Attempts: {attempt_text}",
        )
        return deterministic_report

    def _run_post_mc_hypothesis_recovery(
        self,
        result: RunResult,
        deterministic_report: WaveMonteCarloReport,
        model_name: str,
        timeout_seconds: float,
    ) -> InsightReport | None:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.settings.gcp_region)
        if self.settings.gcp_project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.settings.gcp_project)

        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        recovery_agent = LlmAgent(
            name="PostMonteCarloHypothesisRecoveryAgent",
            model=model_name,
            description="ADK recovery hypothesis writer after Monte Carlo interpretation block.",
            instruction=(
                "You are a coastal hypothesis recovery analyst. Use the provided preliminary hypothesis and Monte Carlo statistics "
                "to produce a rigorous post-simulation hypothesis. Return InsightReport JSON with:\n"
                "- thesis: one precise data-grounded hypothesis\n"
                "- narrative_paragraphs: exactly two paragraphs (first plain-English operational summary, second what changed post-simulation)\n"
                "- evidence_bullets: 4-6 numbered numeric facts\n"
                "- notable_anomalies: 1-4 items\n"
                "- limitations: include uncertainty/model assumptions\n"
                "- recommended_followups: 2-4 concrete actions"
            ),
            tools=[],
            output_schema=InsightReport,
            output_key="insight_report",
        )

        session_service = InMemorySessionService()
        runner = Runner(agent=recovery_agent, app_name=self.settings.app_name, session_service=session_service)
        session_id = f"oceanwatch-mc-recovery-{uuid4().hex[:8]}"

        mc_stats = {
            "final_p10_m": deterministic_report.final_p10_m,
            "final_p50_m": deterministic_report.final_p50_m,
            "final_p90_m": deterministic_report.final_p90_m,
            "probability_exceed_2m": deterministic_report.probability_exceed_2m,
            "probability_exceed_3m": deterministic_report.probability_exceed_3m,
            "probability_reach_3m_anytime": deterministic_report.probability_reach_3m_anytime,
            "cvar95_final_m": deterministic_report.cvar95_final_m,
            "expected_peak_m": deterministic_report.expected_peak_m,
            "drift_per_hour": deterministic_report.drift_per_hour,
            "volatility_per_sqrt_hour": deterministic_report.volatility_per_sqrt_hour,
            "mean_reversion_kappa": deterministic_report.mean_reversion_kappa,
        }

        prompt = (
            f"Station: {result.station.get('display_name', 'selected station')}. "
            f"Question: {result.request.user_question}. "
            f"Preliminary hypothesis: {result.insight.thesis}. "
            f"Monte Carlo stats: {json.dumps(mc_stats)}. "
            "Write the updated post-Monte-Carlo hypothesis with clear numbers and implications."
        )

        async def _drive() -> dict[str, object]:
            await session_service.create_session(
                app_name=self.settings.app_name,
                user_id=result.request.user_id,
                session_id=session_id,
                state={
                    "station_key": result.request.station_key,
                    "mc_stats": mc_stats,
                    "preliminary_hypothesis": result.insight.thesis,
                },
            )

            captured_state: dict[str, object] = {}
            message = Content(role="user", parts=[Part(text=prompt)])
            async for event in runner.run_async(
                user_id=result.request.user_id,
                session_id=session_id,
                new_message=message,
            ):
                actions = getattr(event, "actions", None)
                delta = getattr(actions, "state_delta", None) if actions else None
                if isinstance(delta, dict):
                    captured_state.update(delta)

            session = await session_service.get_session(
                app_name=self.settings.app_name,
                user_id=result.request.user_id,
                session_id=session_id,
            )
            if hasattr(session, "state") and isinstance(session.state, dict):
                captured_state.update(session.state)
            return captured_state

        state = asyncio.run(asyncio.wait_for(_drive(), timeout=timeout_seconds))
        return self._insight_from_state(state)

    def _run_adk_pipeline(self, runtime: ToolRuntime, timeout_seconds: float | None = None) -> None:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.settings.gcp_region)
        if self.settings.gcp_project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.settings.gcp_project)

        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        toolset = OceanWatchToolset(self.settings, self.client, runtime)
        root_agent = create_oceanwatch_root_agent(
            self.settings,
            toolset.tools(),
            model_name=runtime.adk_model_used,
        )

        session_service = InMemorySessionService()
        runner = Runner(agent=root_agent, app_name=self.settings.app_name, session_service=session_service)

        session_id = f"oceanwatch-{uuid4().hex[:10]}"
        prompt = (
            "Analyze coastal conditions for the selected station and produce a rigorous data-grounded hypothesis. "
            f"Station name: {runtime.station.display_name}. "
            f"NDBC station: {runtime.station.ndbc_station_id}. "
            f"CO-OPS station: {runtime.station.coops_station_id}. "
            f"Time horizon (hours): {runtime.request.hours_back}. "
            f"User question: {runtime.request.user_question}"
        )

        async def _drive() -> dict[str, object]:
            await session_service.create_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
                state={
                    "station_key": runtime.station.key,
                    "hours_back": runtime.request.hours_back,
                },
            )

            message = Content(role="user", parts=[Part(text=prompt)])
            captured_state: dict[str, object] = {}

            async for event in runner.run_async(
                user_id=runtime.request.user_id,
                session_id=session_id,
                new_message=message,
            ):
                actions = getattr(event, "actions", None)
                delta = getattr(actions, "state_delta", None) if actions else None
                if isinstance(delta, dict):
                    captured_state.update(delta)

            session = await session_service.get_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
            )
            if hasattr(session, "state") and isinstance(session.state, dict):
                captured_state.update(session.state)

            return captured_state

        timeout_to_use = timeout_seconds or self.settings.adk_timeout_seconds
        _trace(
            runtime,
            "adk.primary_start",
            details=f"model={runtime.adk_model_used or self.settings.vertex_model}, timeout={timeout_to_use:.0f}s",
        )
        started = time.perf_counter()
        try:
            runtime.state_delta = asyncio.run(
                asyncio.wait_for(_drive(), timeout=timeout_to_use)
            )
            _trace(runtime, "adk.primary_success", duration_ms=_elapsed_ms(started))
        except asyncio.TimeoutError as exc:
            _trace(
                runtime,
                "adk.primary_timeout",
                status="error",
                details=f"timeout={timeout_to_use:.0f}s",
                duration_ms=_elapsed_ms(started),
            )
            raise RuntimeError(
                f"ADK primary pipeline timed out after {timeout_to_use:.0f} seconds."
            ) from exc
        
    def _prefetch_sources(self, runtime: ToolRuntime) -> None:
        toolset = OceanWatchToolset(self.settings, self.client, runtime)
        _trace(runtime, "run.prefetch_start", details="parallel NOAA fetch")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(toolset.get_ndbc_observations, runtime.station.ndbc_station_id, runtime.request.hours_back),
                executor.submit(toolset.get_coops_water_level, runtime.station.coops_station_id),
                executor.submit(toolset.get_coops_tide_predictions, runtime.station.coops_station_id),
            ]
            for future in as_completed(futures):
                future.result()

    def _run_adk_recovery(self, runtime: ToolRuntime, timeout_seconds: float | None = None) -> None:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.settings.gcp_region)
        if self.settings.gcp_project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.settings.gcp_project)

        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai.types import Content, Part

        recovery_context = _build_recovery_context(runtime)
        recovery_agent = LlmAgent(
            name="ADKRecoverySynthesisAgent",
            model=runtime.adk_model_used or self.settings.vertex_model,
            description="Ultra-light ADK recovery synthesizer after multi-agent failure.",
            instruction=(
                "You are the ADK recovery synthesizer. "
                "Use ONLY the user message context and return InsightReport JSON fields. "
                "Produce one clear hypothesis backed by explicit evidence and limitations."
            ),
            tools=[],
            output_schema=InsightReport,
            output_key="insight_report",
        )

        session_service = InMemorySessionService()
        runner = Runner(agent=recovery_agent, app_name=self.settings.app_name, session_service=session_service)

        session_id = f"oceanwatch-recovery-{uuid4().hex[:8]}"
        prompt = (
            f"Station: {runtime.station.display_name}. "
            f"Horizon hours: {runtime.request.hours_back}. "
            f"User question: {runtime.request.user_question}. "
            f"Structured context: {json.dumps(recovery_context)}"
        )

        async def _drive() -> dict[str, object]:
            await session_service.create_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
                state={
                    "station_key": runtime.station.key,
                    "hours_back": runtime.request.hours_back,
                },
            )

            captured_state: dict[str, object] = {}
            message = Content(role="user", parts=[Part(text=prompt)])
            async for event in runner.run_async(
                user_id=runtime.request.user_id,
                session_id=session_id,
                new_message=message,
            ):
                actions = getattr(event, "actions", None)
                delta = getattr(actions, "state_delta", None) if actions else None
                if isinstance(delta, dict):
                    captured_state.update(delta)

            session = await session_service.get_session(
                app_name=self.settings.app_name,
                user_id=runtime.request.user_id,
                session_id=session_id,
            )
            if hasattr(session, "state") and isinstance(session.state, dict):
                captured_state.update(session.state)
            return captured_state

        timeout_to_use = timeout_seconds or self.settings.adk_timeout_seconds
        _trace(
            runtime,
            "adk.recovery_start",
            details=f"model={runtime.adk_model_used or self.settings.vertex_model}, timeout={timeout_to_use:.0f}s",
        )
        started = time.perf_counter()
        try:
            runtime.state_delta = asyncio.run(
                asyncio.wait_for(_drive(), timeout=timeout_to_use)
            )
            _trace(runtime, "adk.recovery_success", duration_ms=_elapsed_ms(started))
        except asyncio.TimeoutError as exc:
            _trace(
                runtime,
                "adk.recovery_timeout",
                status="error",
                details=f"timeout={timeout_to_use:.0f}s",
                duration_ms=_elapsed_ms(started),
            )
            raise RuntimeError(
                f"ADK recovery pipeline timed out after {timeout_to_use:.0f} seconds."
            ) from exc

    def _run_adk_with_model_candidates(self, runtime: ToolRuntime) -> None:
        primary_exc: Exception | None = None
        recovery_exc: Exception | None = None
        attempt_summaries: list[str] = []
        primary_model = self._last_successful_model or self.settings.vertex_model
        fallback_models = [m for m in self.settings.vertex_model_candidates if m != primary_model]

        def _attempt_with_model(model_name: str) -> tuple[bool, str | None]:
            nonlocal primary_exc, recovery_exc
            runtime.adk_model_used = model_name
            _trace(
                runtime,
                "adk.model_attempt",
                details=f"model={model_name}, timeout={self.settings.adk_timeout_seconds:.1f}s",
            )
            try:
                self._run_adk_pipeline(runtime, timeout_seconds=self.settings.adk_timeout_seconds)
                runtime.adk_status = "success"
                self._last_successful_model = model_name
                return True, None
            except Exception as exc:
                primary_exc = exc
                details = _flatten_exception_messages(exc)
                category, _ = _classify_adk_error(details)
                runtime.adk_error_category = category
                attempt_summaries.append(f"model={model_name}, primary={category}")
                _trace(runtime, "adk.primary_failed", status="error", details=f"model={model_name}, category={category}")

                if category in {"rate_limit", "network"}:
                    time.sleep(1.0)
                    try:
                        self._run_adk_pipeline(
                            runtime,
                            timeout_seconds=max(20.0, self.settings.adk_timeout_seconds - 5.0),
                        )
                        runtime.adk_status = "success"
                        self._last_successful_model = model_name
                        return True, None
                    except Exception as retry_exc:
                        primary_exc = retry_exc
                        retry_category, _ = _classify_adk_error(_flatten_exception_messages(retry_exc))
                        runtime.adk_error_category = retry_category
                        attempt_summaries.append(f"model={model_name}, retry={retry_category}")
                        _trace(runtime, "adk.primary_retry_failed", status="error", details=f"model={model_name}, category={retry_category}")

                try:
                    self._run_adk_recovery(
                        runtime,
                        timeout_seconds=max(20.0, self.settings.adk_timeout_seconds - 5.0),
                    )
                    runtime.adk_status = "recovery_success"
                    self._last_successful_model = model_name
                    return True, None
                except Exception as recovery_fail_exc:
                    recovery_exc = recovery_fail_exc
                    recovery_category, _ = _classify_adk_error(_flatten_exception_messages(recovery_fail_exc))
                    runtime.adk_error_category = recovery_category
                    attempt_summaries.append(f"model={model_name}, recovery={recovery_category}")
                    _trace(runtime, "adk.recovery_failed", status="error", details=f"model={model_name}, category={recovery_category}")
                    return False, category

        ok, primary_failure_category = _attempt_with_model(primary_model)
        if ok:
            return

        # Try a second model for config-like and ambiguous schema/runtime failures.
        if primary_failure_category in {"model_config", "unknown", "schema"} and fallback_models:
            fallback_model = fallback_models[0]
            _trace(runtime, "adk.switch_model", details=f"from={primary_model} to={fallback_model}")
            ok, _ = _attempt_with_model(fallback_model)
            if ok:
                runtime.warnings.append(f"ADK succeeded using fallback model '{fallback_model}'.")
                return

        runtime.adk_status = "failed_optional"
        runtime.adk_error_summary = self._format_adk_error_summary(primary_exc, recovery_exc)
        runtime.adk_error_category = runtime.adk_error_category or _classify_adk_error(runtime.adk_error_summary)[0]
        if attempt_summaries:
            runtime.adk_error_summary += " Attempts: " + "; ".join(attempt_summaries)
        runtime.warnings.append(runtime.adk_error_summary)
        _trace(runtime, "adk.failed_all_attempts", status="error", details=runtime.adk_error_summary)

    def _collect_direct(self, runtime: ToolRuntime) -> None:
        toolset = OceanWatchToolset(self.settings, self.client, runtime)
        _trace(runtime, "run.collect_direct_start", details="deterministic parallel source collection")

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(toolset.get_ndbc_observations, runtime.station.ndbc_station_id, runtime.request.hours_back),
                executor.submit(toolset.get_coops_water_level, runtime.station.coops_station_id),
                executor.submit(toolset.get_coops_tide_predictions, runtime.station.coops_station_id),
            ]
            for future in as_completed(futures):
                future.result()

        toolset.compute_ocean_metrics(source="all")
        toolset.compute_risk_analytics()
        toolset.simulate_coastal_scenarios()
        toolset.build_visuals()
        _trace(runtime, "run.collect_direct_complete", details="metrics/risk/simulation/visuals complete")

    def _insight_from_state(self, state: dict[str, object]) -> InsightReport | None:
        raw = state.get("insight_report")
        if raw is None:
            return None

        if isinstance(raw, InsightReport):
            return raw

        if isinstance(raw, dict):
            try:
                return InsightReport.model_validate(raw)
            except Exception:
                return None

        if isinstance(raw, str):
            try:
                as_dict = json.loads(raw)
                if isinstance(as_dict, dict):
                    return InsightReport.model_validate(as_dict)
            except Exception:
                return InsightReport(
                    thesis=raw,
                    evidence_bullets=[],
                    notable_anomalies=[],
                    limitations=["Structured schema unavailable from synthesis response."],
                    recommended_followups=[],
                )

        return None

    def _wave_monte_carlo_from_state(self, state: dict[str, object]) -> WaveMonteCarloReport | None:
        raw = state.get("wave_monte_carlo_report")
        if raw is None:
            return None
        if isinstance(raw, WaveMonteCarloReport):
            return raw
        if isinstance(raw, dict):
            try:
                return WaveMonteCarloReport.model_validate(raw)
            except Exception:
                return None
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return None
            if isinstance(parsed, dict):
                try:
                    return WaveMonteCarloReport.model_validate(parsed)
                except Exception:
                    return None
        return None

    def _wave_hypothesis_interpretation_from_state(
        self,
        state: dict[str, object],
    ) -> WaveHypothesisInterpretation | None:
        candidate_keys = [
            "wave_hypothesis_interpretation",
            "mc_hypothesis_interpretation",
            "wave_hypothesis_report",
            "wave_hypothesis",
            "hypothesis",
            "final_output",
            "output",
        ]
        for key in candidate_keys:
            parsed = _coerce_wave_interpretation(state.get(key))
            if parsed is not None:
                return parsed

        for value in state.values():
            parsed = _coerce_wave_interpretation(value)
            if parsed is not None:
                return parsed
        return None

    def _post_monte_carlo_hypothesis_from_state(
        self,
        state: dict[str, object],
    ) -> PostMonteCarloHypothesis | None:
        candidate_keys = [
            "post_mc_hypothesis_report",
            "post_mc_hypothesis",
            "post_monte_carlo_hypothesis",
            "final_output",
            "output",
        ]
        for key in candidate_keys:
            parsed = _coerce_post_mc_hypothesis(state.get(key))
            if parsed is not None:
                return parsed

        for value in state.values():
            parsed = _coerce_post_mc_hypothesis(value)
            if parsed is not None:
                return parsed
        return None

    def _deterministic_wave_monte_carlo_report(
        self,
        deterministic_payload: dict[str, object],
        station_name: str,
        source: str,
    ) -> WaveMonteCarloReport:
        probability_2m = float(deterministic_payload.get("probability_exceed_2m", 0.0) or 0.0)
        probability_3m = float(deterministic_payload.get("probability_exceed_3m", 0.0) or 0.0)
        probability_3m_any = float(deterministic_payload.get("probability_reach_3m_anytime", 0.0) or 0.0)
        p50 = float(deterministic_payload.get("final_p50_m", 0.0) or 0.0)
        p90 = float(deterministic_payload.get("final_p90_m", 0.0) or 0.0)

        risk_label = "high" if probability_3m_any >= 0.35 else "moderate" if probability_2m >= 0.35 else "low"
        thesis = (
            f"Wave-path simulation at {station_name} indicates {risk_label} near-term surf risk: "
            f"median end-state near {p50:.2f} m and 90th-percentile near {p90:.2f} m."
        )
        interpretation_points = [
            f"{probability_2m * 100:.1f}% of simulated paths end above 2.0 m.",
            f"{probability_3m * 100:.1f}% of simulated paths end above 3.0 m.",
            f"{probability_3m_any * 100:.1f}% of paths touch 3.0 m at least once in the horizon.",
            "Use P90 values for conservative planning windows around high tide.",
        ]
        plain_english = (
            f"In plain English: the simulation suggests {risk_label} surf pressure at {station_name} over the next "
            f"{int(deterministic_payload.get('horizon_hours', 0) or 0)} hours. A typical path finishes near {p50:.2f} m, "
            f"but rougher outcomes near {p90:.2f} m are plausible. Treat this as a risk range, not a single forecast line."
        )

        return WaveMonteCarloReport(
            source=source,
            status=str(deterministic_payload.get("status", "ok")),
            thesis=thesis,
            horizon_hours=int(deterministic_payload.get("horizon_hours", 0) or 0),
            path_count=int(deterministic_payload.get("path_count", 0) or 0),
            model_name=str(deterministic_payload.get("model_name", "stochastic_wave_model")),
            latest_wave_height_m=deterministic_payload.get("latest_wave_height_m"),
            drift_per_hour=float(deterministic_payload.get("drift_per_hour", 0.0) or 0.0),
            volatility_per_sqrt_hour=float(deterministic_payload.get("volatility_per_sqrt_hour", 0.0) or 0.0),
            mean_reversion_kappa=float(deterministic_payload.get("mean_reversion_kappa", 0.0) or 0.0),
            long_run_mean_m=float(deterministic_payload.get("long_run_mean_m", 0.0) or 0.0),
            wind_sensitivity_beta=float(deterministic_payload.get("wind_sensitivity_beta", 0.0) or 0.0),
            wind_forcing_component=float(deterministic_payload.get("wind_forcing_component", 0.0) or 0.0),
            jump_intensity_per_hour=float(deterministic_payload.get("jump_intensity_per_hour", 0.0) or 0.0),
            student_t_df=float(deterministic_payload.get("student_t_df", 0.0) or 0.0),
            calm_regime_persistence=float(deterministic_payload.get("calm_regime_persistence", 0.0) or 0.0),
            storm_regime_persistence=float(deterministic_payload.get("storm_regime_persistence", 0.0) or 0.0),
            storm_vol_multiplier=float(deterministic_payload.get("storm_vol_multiplier", 1.0) or 1.0),
            final_p10_m=float(deterministic_payload.get("final_p10_m", 0.0) or 0.0),
            final_p50_m=float(deterministic_payload.get("final_p50_m", 0.0) or 0.0),
            final_p90_m=float(deterministic_payload.get("final_p90_m", 0.0) or 0.0),
            expected_peak_m=float(deterministic_payload.get("expected_peak_m", 0.0) or 0.0),
            cvar95_final_m=float(deterministic_payload.get("cvar95_final_m", 0.0) or 0.0),
            probability_exceed_2m=probability_2m,
            probability_exceed_3m=probability_3m,
            probability_reach_3m_anytime=probability_3m_any,
            plain_english_summary=plain_english,
            mc_adk_status="blocked_required",
            mc_adk_attempts=[],
            mc_hypothesis_ready=False,
            mc_adk_hypothesis=None,
            post_mc_hypothesis=None,
            simulation_stats={
                "risk_label": risk_label,
                "probability_exceed_2m_pct": round(probability_2m * 100.0, 2),
                "probability_exceed_3m_pct": round(probability_3m * 100.0, 2),
                "probability_reach_3m_anytime_pct": round(probability_3m_any * 100.0, 2),
                "final_p10_m": round(float(deterministic_payload.get("final_p10_m", 0.0) or 0.0), 3),
                "final_p50_m": round(float(deterministic_payload.get("final_p50_m", 0.0) or 0.0), 3),
                "final_p90_m": round(float(deterministic_payload.get("final_p90_m", 0.0) or 0.0), 3),
                "expected_peak_m": round(float(deterministic_payload.get("expected_peak_m", 0.0) or 0.0), 3),
                "cvar95_final_m": round(float(deterministic_payload.get("cvar95_final_m", 0.0) or 0.0), 3),
            },
            trajectory_quantiles=list(deterministic_payload.get("trajectory_quantiles", []) or []),
            sample_paths=list(deterministic_payload.get("sample_paths", []) or []),
            final_state_sample=list(deterministic_payload.get("final_state_sample", []) or []),
            interpretation_points=interpretation_points,
            limitations=list(deterministic_payload.get("limitations", []) or []),
        )

    def _monte_carlo_plain_english_from_report(self, report: WaveMonteCarloReport) -> str:
        risk_label = (
            "high"
            if report.probability_reach_3m_anytime >= 0.35
            else "moderate"
            if report.probability_exceed_2m >= 0.35
            else "lower"
        )
        return (
            "In plain English: "
            f"this run suggests {risk_label} near-term surf risk. "
            f"The middle path ends around {report.final_p50_m:.2f} m, while tougher-but-plausible outcomes reach about {report.final_p90_m:.2f} m. "
            f"There is a {report.probability_reach_3m_anytime * 100:.1f}% chance at least one surge above 3.0 m occurs in this horizon."
        )

    def _fallback_insight(self, runtime: ToolRuntime, report) -> InsightReport:
        buoy_lookup = metric_lookup(report.buoy_metrics)
        water_lookup = metric_lookup(report.water_level_metrics)

        wave = buoy_lookup.get("wave_height_m")
        wind = buoy_lookup.get("wind_speed_mps")
        level = water_lookup.get("water_level_m")
        severity = report.advanced_analytics.get("severity", {})

        thesis = (
            f"At {runtime.station.display_name}, modeled severity is "
            f"{severity.get('level', 'Unknown')} ({severity.get('score_0_100', 'n/a')}/100), "
            "with short-term ocean variability that merits monitoring."
        )

        evidence = []
        if wave and wave.latest is not None:
            evidence.append(
                f"Latest wave height is {wave.latest:.2f} m with {wave.anomaly_count} z-score anomalies."
            )
        if wind and wind.latest is not None:
            evidence.append(
                f"Latest wind speed is {wind.latest:.2f} m/s and trend slope is {(wind.slope_per_hour or 0):.3f} per hour."
            )
        if level and level.latest is not None:
            evidence.append(
                f"Latest observed water level is {level.latest:.2f} m with slope {(level.slope_per_hour or 0):.3f} per hour."
            )

        anomalies = []
        for item in report.advanced_analytics.get("irregular_patterns", [])[:5]:
            anomalies.append(
                f"{item.get('label', item.get('field', 'series'))}: {item.get('interpretation', 'irregular movement')}"
            )

        limitations = []
        for source, status in runtime.health.items():
            if status != "ok":
                limitations.append(f"{source} status is '{status}', reducing confidence.")
        if not limitations:
            limitations.append("All configured sources responded successfully for this run.")

        followups = [
            "Re-run with a shorter 24-48 hour horizon to isolate event-driven shifts.",
            "Compare with a second nearby station to separate local and regional signals.",
            "Use simulation outcomes to choose a low, medium, and high preparedness plan.",
        ]

        return InsightReport(
            thesis=thesis,
            narrative_paragraphs=[
                (
                    f"Current conditions at {runtime.station.display_name} indicate a {severity.get('level', 'moderate').lower()} "
                    "coastal-stress regime. The signal is not driven by one variable alone; instead, wave behavior, wind forcing, "
                    "and observed water-level movement collectively suggest elevated short-term variability."
                ),
                (
                    "The strongest evidence is in the directional trends and anomaly density across the selected horizon. "
                    "When wave and wind signals align while water-level slope remains non-neutral, the probability of "
                    "near-term coastal pressure events increases, even if absolute magnitudes are not yet extreme."
                ),
            ],
            evidence_bullets=evidence,
            notable_anomalies=anomalies,
            limitations=limitations,
            recommended_followups=followups,
        )

    def _adk_failure_insight(self, runtime: ToolRuntime) -> InsightReport:
        error_text = runtime.adk_error_summary or "ADK execution failed."
        return InsightReport(
            thesis=(
                "ADK-required synthesis did not complete, so this run cannot return a grading-valid hypothesis statement."
            ),
            narrative_paragraphs=[
                "Preliminary runtime collection and exploratory analytics completed successfully, but the ADK synthesis stage did not finalize.",
                "This means we can still inspect data patterns and risk indicators, yet the final agent-generated hypothesis narrative is pending a successful ADK/Vertex run.",
            ],
            evidence_bullets=[
                "Runtime NOAA collection and EDA still executed for observability.",
                "Final hypothesis output is blocked until ADK/Vertex run succeeds.",
            ],
            notable_anomalies=[],
            limitations=[error_text],
            recommended_followups=[
                "Retry after confirming Vertex quota/credentials/network status.",
                "Set OCEANWATCH_REQUIRE_ADK_SUCCESS=false only for non-graded demos.",
            ],
        )

    def _format_adk_error_summary(self, primary_exc: Exception | None, recovery_exc: Exception | None) -> str:
        primary_details = _flatten_exception_messages(primary_exc) if primary_exc else ""
        recovery_details = _flatten_exception_messages(recovery_exc) if recovery_exc else ""
        details = " | ".join(item for item in [primary_details, recovery_details] if item)
        if len(details) > 320:
            details = details[:320] + "..."
        category, action = _classify_adk_error(details)
        return (
            f"ADK execution failed ({category}). {action} "
            f"Technical summary: {details or 'No detailed exception payload.'}"
        )


def _flatten_exception_messages(exc: Exception | None) -> str:
    if exc is None:
        return ""
    if isinstance(exc, ExceptionGroup):
        nested = [_flatten_exception_messages(item) for item in exc.exceptions]
        merged = "; ".join(item for item in nested if item)
        base = merged or str(exc) or repr(exc)
        return _sanitize_exception_text(base)
    base = str(exc).strip()
    if not base:
        base = repr(exc)
    return _sanitize_exception_text(base)


def _classify_adk_error(details: str) -> tuple[str, str]:
    text = (details or "").lower()
    if any(
        token in text
        for token in [
            "validationerror",
            "pydantic",
            "missing wavehypothesisinterpretation",
            "missing postmontecarlohypothesis",
            "missing post_monte_carlo",
            "structured output",
            "schema",
        ]
    ):
        return (
            "schema",
            "ADK returned an unexpected response shape. Retry once; if it persists, simplify prompt/schema.",
        )
    if any(
        token in text
        for token in [
            "defaultcredentialserror",
            "application default credentials",
            "could not automatically determine credentials",
            "project was not passed",
            "google_cloud_project",
        ]
    ):
        return (
            "config",
            "Local Vertex config is incomplete. Set GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION and run gcloud auth application-default login.",
        )
    if any(
        token in text
        for token in [
            "model not found",
            "unsupported model",
            "invalid model",
            "invalid argument",
            "not found: publisher model",
        ]
    ):
        return (
            "model_config",
            "The configured Vertex model may be unavailable in this region/project. Try fallback model candidates.",
        )
    if any(token in text for token in ["429", "resource_exhausted", "quota", "rate limit"]):
        return (
            "rate_limit",
            "Vertex quota/rate limit was hit. Wait 1-2 minutes or switch to a lighter model.",
        )
    if any(token in text for token in ["403", "permission", "credential", "auth", "unauthorized"]):
        return (
            "auth",
            "Authentication or permissions are missing. Check gcloud auth and Vertex IAM access.",
        )
    if any(token in text for token in ["timeout", "timed out", "connection", "dns", "network", "unavailable"]):
        return (
            "network",
            "Network/endpoint instability interrupted ADK. Retry with stable connectivity.",
        )
    return (
        "unknown",
        "An unexpected ADK runtime error occurred. Review logs and retry.",
    )


def _sanitize_exception_text(message: str) -> str:
    sanitized = message.strip()
    sanitized = sanitized.replace(
        "unhandled errors in a TaskGroup (1 sub-exception)",
        "parallel ADK sub-agent execution failed",
    )
    sanitized = sanitized.replace(
        "unhandled errors in a TaskGroup",
        "parallel ADK sub-agent execution failed",
    )
    return sanitized


def _coerce_wave_interpretation(raw: object) -> WaveHypothesisInterpretation | None:
    if raw is None:
        return None
    if isinstance(raw, WaveHypothesisInterpretation):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
        if "mc_adk_hypothesis" in payload and "plain_english_summary" in payload:
            try:
                return WaveHypothesisInterpretation.model_validate(payload)
            except Exception:
                return None
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return WaveHypothesisInterpretation(
                mc_adk_hypothesis=text,
                plain_english_summary=text,
                interpretation_points=[],
                limitations=[],
            )
        if isinstance(parsed, Mapping):
            return _coerce_wave_interpretation(dict(parsed))
    return None


def _coerce_post_mc_hypothesis(raw: object) -> PostMonteCarloHypothesis | None:
    if raw is None:
        return None
    if isinstance(raw, PostMonteCarloHypothesis):
        return raw
    if isinstance(raw, Mapping):
        payload = dict(raw)
        if "post_mc_hypothesis" in payload:
            try:
                return PostMonteCarloHypothesis.model_validate(payload)
            except Exception:
                return None
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return PostMonteCarloHypothesis(post_mc_hypothesis=text, what_changed=[])
        if isinstance(parsed, Mapping):
            return _coerce_post_mc_hypothesis(dict(parsed))
    return None


def _unavailable_payload(source_name: str, station_id: str, error: str) -> SourcePayload:
    return SourcePayload(
        source_name=source_name,
        station_id=station_id,
        available=False,
        row_count=0,
        records=[],
        metadata={"generated_at_utc": datetime.utcnow().isoformat()},
        error=error,
    )


def _build_recovery_context(runtime: ToolRuntime) -> dict[str, object]:
    report = runtime.cached_metrics_report or compute_ocean_metrics(
        runtime.buoy_frame,
        runtime.water_level_frame,
        runtime.tide_prediction_frame,
    )
    advanced = report.advanced_analytics or {}
    return {
        "health": runtime.health,
        "metric_cards": report.metric_cards,
        "confidence_note": report.confidence_note,
        "severity": advanced.get("severity", {}),
        "regression_signals": advanced.get("regression_signals", [])[:3],
        "irregular_patterns": advanced.get("irregular_patterns", [])[:5],
        "simulation_outcomes": advanced.get("simulation_outcomes", [])[:3],
        "warnings": runtime.warnings[-3:],
    }


def _trace(
    runtime: ToolRuntime,
    step: str,
    status: str = "ok",
    details: str | None = None,
    duration_ms: int | None = None,
) -> None:
    entry: dict[str, object] = {
        "ts_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "step": step,
        "status": status,
    }
    if details:
        entry["details"] = details
    if duration_ms is not None:
        entry["duration_ms"] = duration_ms
    runtime.execution_trace.append(entry)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _tool_payload_brief(payload: SourcePayload, sample_rows: int = 12) -> dict[str, object]:
    return {
        "source_name": payload.source_name,
        "station_id": payload.station_id,
        "available": payload.available,
        "row_count": payload.row_count,
        "fetched_at_utc": payload.fetched_at_utc,
        "metadata": payload.metadata,
        "error": payload.error,
        "sample_records": payload.records[-sample_rows:],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class AnalysisRequest(BaseModel):
    user_question: str = Field(min_length=5)
    station_key: str
    hours_back: int = Field(default=168, ge=24, le=336)
    units: str = Field(default="metric")
    analysis_focus: str | None = None
    user_id: str = Field(default="local-user")


class SourcePayload(BaseModel):
    source_name: str
    station_id: str
    available: bool
    row_count: int = 0
    records: list[dict[str, Any]] = Field(default_factory=list)
    fetched_at_utc: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class SeriesMetrics(BaseModel):
    field: str
    count: int
    mean: float | None
    median: float | None
    min: float | None
    max: float | None
    latest: float | None
    slope_per_hour: float | None
    anomaly_count: int


class MetricsReport(BaseModel):
    metric_cards: dict[str, float | int | str | None] = Field(default_factory=dict)
    buoy_metrics: list[SeriesMetrics] = Field(default_factory=list)
    water_level_metrics: list[SeriesMetrics] = Field(default_factory=list)
    tide_metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    advanced_analytics: dict[str, Any] = Field(default_factory=dict)
    confidence_note: str = ""


class InsightReport(BaseModel):
    thesis: str = Field(validation_alias=AliasChoices("thesis", "hypothesis"))
    narrative_paragraphs: list[str] = Field(default_factory=list)
    evidence_bullets: list[str] = Field(default_factory=list)
    notable_anomalies: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_followups: list[str] = Field(default_factory=list)


class BuoyAgentReport(BaseModel):
    status: str
    highlights: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class TideAgentReport(BaseModel):
    status: str
    highlights: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class RiskAgentReport(BaseModel):
    status: str
    severity_level: str
    likelihood_summary: str
    highlights: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class WaveMonteCarloReport(BaseModel):
    source: str = "deterministic"
    status: str = "ok"
    thesis: str = Field(validation_alias=AliasChoices("thesis", "hypothesis"))
    horizon_hours: int
    path_count: int
    model_name: str = "stochastic_wave_model"
    latest_wave_height_m: float | None = None
    drift_per_hour: float = 0.0
    volatility_per_sqrt_hour: float = 0.0
    mean_reversion_kappa: float = 0.0
    long_run_mean_m: float = 0.0
    wind_sensitivity_beta: float = 0.0
    wind_forcing_component: float = 0.0
    jump_intensity_per_hour: float = 0.0
    student_t_df: float = 0.0
    calm_regime_persistence: float = 0.0
    storm_regime_persistence: float = 0.0
    storm_vol_multiplier: float = 1.0
    final_p10_m: float = 0.0
    final_p50_m: float = 0.0
    final_p90_m: float = 0.0
    expected_peak_m: float = 0.0
    cvar95_final_m: float = 0.0
    probability_exceed_2m: float = 0.0
    probability_exceed_3m: float = 0.0
    probability_reach_3m_anytime: float = 0.0
    plain_english_summary: str = ""
    mc_adk_status: str = "blocked_required"
    mc_adk_attempts: list[str] = Field(default_factory=list)
    mc_hypothesis_ready: bool = False
    mc_adk_hypothesis: str | None = None
    post_mc_hypothesis: str | None = None
    simulation_stats: dict[str, Any] = Field(default_factory=dict)
    trajectory_quantiles: list[dict[str, float]] = Field(default_factory=list)
    sample_paths: list[list[float]] = Field(default_factory=list)
    final_state_sample: list[float] = Field(default_factory=list)
    interpretation_points: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class WaveSimulationSnapshot(BaseModel):
    station_name: str
    horizon_hours: int
    path_count: int
    model_name: str
    final_p50_m: float
    final_p90_m: float
    expected_peak_m: float
    cvar95_final_m: float
    probability_exceed_2m: float
    probability_exceed_3m: float
    probability_reach_3m_anytime: float
    mean_reversion_kappa: float
    volatility_per_sqrt_hour: float
    jump_intensity_per_hour: float
    limitations: list[str] = Field(default_factory=list)


class WaveHypothesisInterpretation(BaseModel):
    mc_adk_hypothesis: str
    plain_english_summary: str
    interpretation_points: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PostMonteCarloHypothesis(BaseModel):
    post_mc_hypothesis: str
    what_changed: list[str] = Field(default_factory=list)


@dataclass
class RunResult:
    request: AnalysisRequest
    station: dict[str, Any]
    health: dict[str, str]
    source_payloads: dict[str, SourcePayload]
    metrics: MetricsReport
    insight: InsightReport
    figures: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    adk_status: str = "not_attempted"
    adk_attempted: bool = False
    adk_error_summary: str | None = None
    adk_model_used: str | None = None
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float = 0.0

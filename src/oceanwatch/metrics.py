from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from oceanwatch.schemas import MetricsReport, SeriesMetrics


def compute_series_metrics(frame: pd.DataFrame, field: str) -> SeriesMetrics:
    if frame.empty or field not in frame.columns:
        return SeriesMetrics(
            field=field,
            count=0,
            mean=None,
            median=None,
            min=None,
            max=None,
            latest=None,
            slope_per_hour=None,
            anomaly_count=0,
        )

    data = frame[["timestamp", field]].dropna().copy()
    if data.empty:
        return SeriesMetrics(
            field=field,
            count=0,
            mean=None,
            median=None,
            min=None,
            max=None,
            latest=None,
            slope_per_hour=None,
            anomaly_count=0,
        )

    values = data[field].astype(float)
    slope = _slope_per_hour(data["timestamp"], values)
    anomaly_count = _zscore_anomaly_count(values)

    return SeriesMetrics(
        field=field,
        count=int(values.count()),
        mean=float(values.mean()),
        median=float(values.median()),
        min=float(values.min()),
        max=float(values.max()),
        latest=float(values.iloc[-1]),
        slope_per_hour=slope,
        anomaly_count=anomaly_count,
    )


def compute_ocean_metrics(
    buoy_frame: pd.DataFrame,
    water_level_frame: pd.DataFrame,
    tide_prediction_frame: pd.DataFrame,
) -> MetricsReport:
    buoy_fields = ["wave_height_m", "wind_speed_mps", "gust_mps", "water_temp_c"]
    water_fields = ["water_level_m"]

    buoy_metrics = [compute_series_metrics(buoy_frame, field) for field in buoy_fields]
    water_metrics = [compute_series_metrics(water_level_frame, field) for field in water_fields]

    if tide_prediction_frame.empty or "tide_type" not in tide_prediction_frame.columns:
        high_tides = pd.DataFrame(columns=["predicted_tide_ft"])
        low_tides = pd.DataFrame(columns=["predicted_tide_ft"])
    else:
        high_tides = tide_prediction_frame[tide_prediction_frame["tide_type"] == "H"]
        low_tides = tide_prediction_frame[tide_prediction_frame["tide_type"] == "L"]

    tide_range = None
    if not high_tides.empty and not low_tides.empty:
        tide_range = float(high_tides["predicted_tide_ft"].max() - low_tides["predicted_tide_ft"].min())

    advanced = _build_advanced_analytics(
        buoy_frame=buoy_frame,
        water_level_frame=water_level_frame,
        tide_prediction_frame=tide_prediction_frame,
        buoy_metrics=buoy_metrics,
        water_metrics=water_metrics,
        tide_range=tide_range,
    )

    metric_cards = {
        "latest_wave_height_m": _latest_value(buoy_frame, "wave_height_m"),
        "latest_wind_speed_mps": _latest_value(buoy_frame, "wind_speed_mps"),
        "latest_water_temp_c": _latest_value(buoy_frame, "water_temp_c"),
        "latest_water_level_m": _latest_value(water_level_frame, "water_level_m"),
        "predicted_tide_range_ft": tide_range,
        "buoy_rows": int(len(buoy_frame)),
        "water_level_rows": int(len(water_level_frame)),
        "prediction_rows": int(len(tide_prediction_frame)),
        "severity_score_0_100": advanced["severity"]["score_0_100"],
        "severity_level": advanced["severity"]["level"],
    }

    confidence_note = _build_confidence_note(
        has_buoy=not buoy_frame.empty,
        has_water=not water_level_frame.empty,
        has_predictions=not tide_prediction_frame.empty,
    )

    tide_metrics = {
        "high_tide_count": int(len(high_tides)),
        "low_tide_count": int(len(low_tides)),
        "predicted_tide_range_ft": tide_range,
    }

    return MetricsReport(
        metric_cards=metric_cards,
        buoy_metrics=buoy_metrics,
        water_level_metrics=water_metrics,
        tide_metrics=tide_metrics,
        advanced_analytics=advanced,
        confidence_note=confidence_note,
    )


def metric_lookup(metrics: Iterable[SeriesMetrics]) -> dict[str, SeriesMetrics]:
    return {entry.field: entry for entry in metrics}


def _latest_value(frame: pd.DataFrame, field: str) -> float | None:
    if frame.empty or field not in frame.columns:
        return None
    filtered = frame[field].dropna()
    if filtered.empty:
        return None
    return float(filtered.iloc[-1])


def _slope_per_hour(timestamps: pd.Series, values: pd.Series) -> float | None:
    if len(values) < 3:
        return None

    ts_ns = timestamps.astype("int64")
    hours = (ts_ns - ts_ns.iloc[0]) / 3_600_000_000_000
    if len(set(hours)) < 2:
        return None

    slope, _ = np.polyfit(hours, values, 1)
    if math.isnan(slope):
        return None
    return float(slope)


def _zscore_anomaly_count(values: pd.Series) -> int:
    if len(values) < 5:
        return 0

    std = float(values.std(ddof=0))
    if std == 0:
        return 0

    mean = float(values.mean())
    zscores = (values - mean) / std
    return int((np.abs(zscores) > 2.0).sum())


def _build_confidence_note(has_buoy: bool, has_water: bool, has_predictions: bool) -> str:
    if has_buoy and has_water and has_predictions:
        return "High confidence: all NOAA sources responded for this run."
    if has_buoy and has_water:
        return "Moderate confidence: tide predictions missing, but buoy and observed water level are available."
    if has_buoy or has_water or has_predictions:
        return "Reduced confidence: one or more data feeds were unavailable during runtime collection."
    return "Low confidence: no data source returned usable rows."


def _build_advanced_analytics(
    buoy_frame: pd.DataFrame,
    water_level_frame: pd.DataFrame,
    tide_prediction_frame: pd.DataFrame,
    buoy_metrics: list[SeriesMetrics],
    water_metrics: list[SeriesMetrics],
    tide_range: float | None,
) -> dict[str, object]:
    metric_map = metric_lookup(buoy_metrics + water_metrics)
    wave_latest = _safe_metric_latest(metric_map.get("wave_height_m"))
    wind_latest = _safe_metric_latest(metric_map.get("wind_speed_mps"))
    level_latest = _safe_metric_latest(metric_map.get("water_level_m"))

    severity = _severity_profile(
        wave_latest=wave_latest,
        wind_latest=wind_latest,
        level_latest=level_latest,
        anomaly_counts=[m.anomaly_count for m in (buoy_metrics + water_metrics)],
        tide_range=tide_range,
    )

    regressions = [
        _build_regression_signal(buoy_frame, "wave_height_m", "Wave Height", threshold=2.4, units="m"),
        _build_regression_signal(buoy_frame, "wind_speed_mps", "Wind Speed", threshold=11.0, units="m/s"),
        _build_regression_signal(water_level_frame, "water_level_m", "Water Level", threshold=1.1, units="m"),
    ]
    regressions = [item for item in regressions if item]

    distributions = [
        _distribution_profile(buoy_frame, "wave_height_m", "Wave Height", "m"),
        _distribution_profile(buoy_frame, "wind_speed_mps", "Wind Speed", "m/s"),
        _distribution_profile(buoy_frame, "water_temp_c", "Water Temperature", "C"),
        _distribution_profile(water_level_frame, "water_level_m", "Water Level", "m"),
    ]
    distributions = [item for item in distributions if item]

    irregular_patterns = []
    irregular_patterns.extend(_irregular_pattern_records(buoy_frame, "wave_height_m", "Wave Height", "m"))
    irregular_patterns.extend(_irregular_pattern_records(buoy_frame, "wind_speed_mps", "Wind Speed", "m/s"))
    irregular_patterns.extend(_irregular_pattern_records(water_level_frame, "water_level_m", "Water Level", "m"))
    irregular_patterns = sorted(irregular_patterns, key=lambda row: abs(float(row["z_score"])), reverse=True)[:8]

    simulation_outcomes = _simulation_outcomes(
        wave_latest=wave_latest,
        wind_latest=wind_latest,
        level_latest=level_latest,
        regressions=regressions,
    )
    wave_outlook = _wave_outlook(metric_map, regressions, distributions)

    risk_headline = (
        f"Severity {severity['level']} ({severity['score_0_100']}/100): "
        f"{severity['summary']}"
    )

    return {
        "severity": severity,
        "regression_signals": regressions,
        "distribution_profiles": distributions,
        "irregular_patterns": irregular_patterns,
        "simulation_outcomes": simulation_outcomes,
        "wave_outlook": wave_outlook,
        "risk_headline": risk_headline,
    }


def _safe_metric_latest(metric: SeriesMetrics | None) -> float:
    if metric and metric.latest is not None:
        return float(metric.latest)
    return 0.0


def _severity_profile(
    wave_latest: float,
    wind_latest: float,
    level_latest: float,
    anomaly_counts: list[int],
    tide_range: float | None,
) -> dict[str, object]:
    wave_score = min(100.0, max(0.0, (wave_latest / 3.2) * 100.0))
    wind_score = min(100.0, max(0.0, (wind_latest / 18.0) * 100.0))
    level_score = min(100.0, max(0.0, (abs(level_latest) / 1.4) * 100.0))
    anomaly_score = min(100.0, (sum(anomaly_counts) / 24.0) * 100.0)
    tide_range_score = min(100.0, max(0.0, ((tide_range or 0.0) / 8.0) * 100.0))

    total = (0.28 * wave_score) + (0.26 * wind_score) + (0.18 * level_score) + (0.18 * anomaly_score) + (
        0.10 * tide_range_score
    )
    score = int(round(total))
    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Moderate"
    else:
        level = "Low"

    drivers = [
        {"name": "Wave Pressure", "score": round(wave_score, 1)},
        {"name": "Wind Pressure", "score": round(wind_score, 1)},
        {"name": "Water-Level Pressure", "score": round(level_score, 1)},
        {"name": "Anomaly Pressure", "score": round(anomaly_score, 1)},
        {"name": "Tide-Range Pressure", "score": round(tide_range_score, 1)},
    ]
    top_drivers = sorted(drivers, key=lambda item: item["score"], reverse=True)[:2]
    summary = (
        f"Top drivers are {top_drivers[0]['name']} ({top_drivers[0]['score']:.1f}) and "
        f"{top_drivers[1]['name']} ({top_drivers[1]['score']:.1f})."
    )
    return {
        "score_0_100": score,
        "level": level,
        "drivers": drivers,
        "summary": summary,
    }


def _build_regression_signal(
    frame: pd.DataFrame,
    field: str,
    label: str,
    threshold: float,
    units: str,
) -> dict[str, object] | None:
    if frame.empty or field not in frame.columns:
        return None

    data = frame[["timestamp", field]].dropna().copy()
    if len(data) < 8:
        return None

    ts_ns = data["timestamp"].astype("int64")
    hours = ((ts_ns - ts_ns.iloc[0]) / 3_600_000_000_000).astype(float)
    if hours.nunique() < 2:
        return None

    values = data[field].astype(float)
    regression = stats.linregress(hours, values)
    projected_6h = float(regression.intercept + regression.slope * (hours.iloc[-1] + 6.0))

    residuals = values - (regression.intercept + regression.slope * hours)
    sigma = float(residuals.std(ddof=0))
    if sigma <= 0:
        sigma = max(float(values.std(ddof=0)), 0.05)

    exceedance_probability = float(1.0 - stats.norm.cdf(threshold, loc=projected_6h, scale=sigma))
    exceedance_probability = float(min(1.0, max(0.0, exceedance_probability)))
    risk_band = "elevated" if exceedance_probability >= 0.6 else "watch" if exceedance_probability >= 0.3 else "low"

    return {
        "field": field,
        "label": label,
        "slope_per_hour": round(float(regression.slope), 4),
        "r_squared": round(float(regression.rvalue**2), 3),
        "p_value": round(float(regression.pvalue), 4),
        "projected_6h": round(projected_6h, 3),
        "threshold": threshold,
        "threshold_units": units,
        "exceedance_probability": round(exceedance_probability, 3),
        "risk_band": risk_band,
    }


def _distribution_profile(frame: pd.DataFrame, field: str, label: str, units: str) -> dict[str, object] | None:
    if frame.empty or field not in frame.columns:
        return None
    values = pd.to_numeric(frame[field], errors="coerce").dropna()
    if len(values) < 6:
        return None
    return {
        "field": field,
        "label": label,
        "units": units,
        "count": int(len(values)),
        "mean": round(float(values.mean()), 3),
        "std": round(float(values.std(ddof=0)), 3),
        "p10": round(float(values.quantile(0.10)), 3),
        "p25": round(float(values.quantile(0.25)), 3),
        "p50": round(float(values.quantile(0.50)), 3),
        "p75": round(float(values.quantile(0.75)), 3),
        "p90": round(float(values.quantile(0.90)), 3),
    }


def _irregular_pattern_records(
    frame: pd.DataFrame,
    field: str,
    label: str,
    units: str,
) -> list[dict[str, object]]:
    if frame.empty or field not in frame.columns:
        return []
    data = frame[["timestamp", field]].dropna().copy()
    if len(data) < 8:
        return []

    values = data[field].astype(float)
    std = float(values.std(ddof=0))
    if std == 0:
        return []
    zscores = (values - float(values.mean())) / std
    flagged = data.loc[zscores.abs() >= 2.2].copy()
    if flagged.empty:
        return []

    flagged["z"] = zscores[flagged.index]
    records = []
    for _, row in flagged.head(4).iterrows():
        z_score = float(row["z"])
        signal = "spike" if z_score > 0 else "drop"
        records.append(
            {
                "field": field,
                "label": label,
                "timestamp_utc": row["timestamp"].isoformat(),
                "value": round(float(row[field]), 3),
                "units": units,
                "z_score": round(z_score, 3),
                "interpretation": f"Unusual {signal} versus recent baseline.",
            }
        )
    return records


def _simulation_outcomes(
    wave_latest: float,
    wind_latest: float,
    level_latest: float,
    regressions: list[dict[str, object]],
) -> list[dict[str, object]]:
    exceedance_lookup = {item["field"]: float(item["exceedance_probability"]) for item in regressions}
    wave_prob = exceedance_lookup.get("wave_height_m", 0.2)
    wind_prob = exceedance_lookup.get("wind_speed_mps", 0.2)
    level_prob = exceedance_lookup.get("water_level_m", 0.2)

    scenarios = [
        ("Baseline Drift", 1.0, 1.0, 0.0),
        ("Storm Pulse (+30%)", 1.30, 1.30, 0.18),
        ("Calm Window (-20%)", 0.80, 0.75, -0.08),
    ]

    outcomes: list[dict[str, object]] = []
    for name, wave_mult, wind_mult, level_shift in scenarios:
        sim_wave = max(0.0, wave_latest * wave_mult)
        sim_wind = max(0.0, wind_latest * wind_mult)
        sim_level = level_latest + level_shift

        score = _scenario_risk_score(sim_wave, sim_wind, sim_level)
        likelihood = float(min(0.99, max(0.01, (wave_prob + wind_prob + level_prob) / 3.0 * wave_mult)))
        level = "High" if score >= 70 else "Moderate" if score >= 40 else "Low"

        outcomes.append(
            {
                "scenario": name,
                "projected_wave_height_m": round(sim_wave, 3),
                "projected_wind_speed_mps": round(sim_wind, 3),
                "projected_water_level_m": round(sim_level, 3),
                "severity_score_0_100": int(round(score)),
                "likelihood_probability": round(likelihood, 3),
                "likelihood_level": level,
                "interpretation": (
                    f"{name} suggests {level.lower()} coastal stress with "
                    f"{likelihood*100:.1f}% modeled likelihood."
                ),
            }
        )
    return outcomes


def _scenario_risk_score(wave_height_m: float, wind_speed_mps: float, water_level_m: float) -> float:
    wave_score = min(100.0, (wave_height_m / 3.2) * 100.0)
    wind_score = min(100.0, (wind_speed_mps / 18.0) * 100.0)
    level_score = min(100.0, (abs(water_level_m) / 1.4) * 100.0)
    return (0.42 * wave_score) + (0.33 * wind_score) + (0.25 * level_score)


def _wave_outlook(
    metric_map: dict[str, SeriesMetrics],
    regressions: list[dict[str, object]],
    distributions: list[dict[str, object]],
) -> dict[str, object]:
    wave_metric = metric_map.get("wave_height_m")
    if wave_metric is None or wave_metric.latest is None:
        return {
            "trend_label": "unknown",
            "estimated_24h_low_m": None,
            "estimated_24h_high_m": None,
            "guidance": ["Wave outlook is unavailable because the buoy wave series has insufficient samples."],
        }

    latest = float(wave_metric.latest)
    slope = float(wave_metric.slope_per_hour or 0.0)
    trend_label = "rising" if slope > 0.03 else "falling" if slope < -0.03 else "stable"

    wave_reg = next((row for row in regressions if row.get("field") == "wave_height_m"), None)
    exceedance_probability = float(wave_reg.get("exceedance_probability", 0.0)) if wave_reg else 0.0

    wave_distribution = next((row for row in distributions if row.get("field") == "wave_height_m"), None)
    std = float(wave_distribution.get("std", 0.15)) if wave_distribution else 0.15
    std = max(std, 0.08)

    projected_mid = max(0.0, latest + (slope * 24.0))
    projected_low = max(0.0, projected_mid - (1.5 * std * math.sqrt(24.0)))
    projected_high = max(projected_low, projected_mid + (1.5 * std * math.sqrt(24.0)))

    guidance = [
        f"Wave trend appears {trend_label}, with slope {slope:+.3f} m/hour.",
        f"24h envelope estimate is {projected_low:.2f} m to {projected_high:.2f} m.",
        f"Probability of exceeding 2.4 m in the next 6h is {exceedance_probability * 100:.1f}%.",
    ]
    if projected_high >= 2.5:
        guidance.append("High-wave regime possible; shoreline operations should prepare for elevated surf.")
    elif projected_high >= 1.5:
        guidance.append("Moderate wave regime likely; monitor shifts around tide turns.")
    else:
        guidance.append("Low-to-moderate wave regime likely if wind forcing remains steady.")

    return {
        "trend_label": trend_label,
        "estimated_24h_low_m": round(projected_low, 3),
        "estimated_24h_high_m": round(projected_high, 3),
        "guidance": guidance,
        "latest_m": round(latest, 3),
        "slope_per_hour": round(slope, 4),
    }


def compute_wave_monte_carlo(
    buoy_frame: pd.DataFrame,
    days_ahead: int = 3,
    path_count: int = 800,
    seed: int = 42,
) -> dict[str, object]:
    horizon_hours = int(max(24, min(24 * 7, days_ahead * 24)))
    path_count = int(max(200, min(4000, path_count)))

    wave_series, wind_series = _hourly_wave_and_wind_series(buoy_frame)
    if wave_series.size < 12:
        return {
            "status": "insufficient_data",
            "model_name": "stochastic_wave_model",
            "horizon_hours": horizon_hours,
            "path_count": path_count,
            "latest_wave_height_m": None,
            "drift_per_hour": 0.0,
            "volatility_per_sqrt_hour": 0.0,
            "mean_reversion_kappa": 0.0,
            "long_run_mean_m": 0.0,
            "wind_sensitivity_beta": 0.0,
            "wind_forcing_component": 0.0,
            "jump_intensity_per_hour": 0.0,
            "student_t_df": 0.0,
            "calm_regime_persistence": 0.0,
            "storm_regime_persistence": 0.0,
            "storm_vol_multiplier": 1.0,
            "final_p10_m": 0.0,
            "final_p50_m": 0.0,
            "final_p90_m": 0.0,
            "expected_peak_m": 0.0,
            "cvar95_final_m": 0.0,
            "probability_exceed_2m": 0.0,
            "probability_exceed_3m": 0.0,
            "probability_reach_3m_anytime": 0.0,
            "trajectory_quantiles": [],
            "sample_paths": [],
            "final_state_sample": [],
            "limitations": ["Not enough buoy wave observations for Monte Carlo simulation."],
        }

    latest = float(wave_series[-1])
    long_run_mean = float(np.mean(wave_series))
    drift_trend = float(np.mean(np.diff(wave_series[-min(24, wave_series.size - 1) :])))

    mu_ou, kappa, residuals = _estimate_ou_parameters(wave_series)
    long_run_mean = float(mu_ou)
    sigma_base = float(np.std(residuals, ddof=0))
    sigma_base = max(sigma_base, 0.05)

    jump_threshold = max(2.5 * sigma_base, 0.12)
    jump_mask = np.abs(residuals) >= jump_threshold
    jump_pool = residuals[jump_mask]
    jump_intensity = float(np.clip(np.mean(jump_mask), 0.002, 0.08))
    jump_mean = float(np.mean(jump_pool)) if jump_pool.size else 0.0
    jump_std = float(np.std(jump_pool, ddof=0)) if jump_pool.size else max(2.0 * sigma_base, 0.08)
    jump_std = max(jump_std, 0.05)

    student_t_df = _estimate_student_t_df(residuals)
    t_scale = math.sqrt(student_t_df / max(student_t_df - 2.0, 1e-6))

    (
        calm_persistence,
        storm_persistence,
        storm_vol_multiplier,
        initial_regime,
    ) = _estimate_regime_parameters(np.diff(wave_series), sigma_base)

    wind_beta = 0.0
    wind_force = 0.0
    if wind_series is not None and wind_series.size == wave_series.size:
        wave_delta = np.diff(wave_series)
        wind_anom = wind_series[:-1] - float(np.mean(wind_series[:-1]))
        if wave_delta.size >= 8 and np.std(wind_anom) > 1e-6:
            x = np.column_stack([np.ones_like(wind_anom), wind_anom])
            coeff = np.linalg.lstsq(x, wave_delta, rcond=None)[0]
            wind_beta = float(coeff[1])
            wind_force = float(wind_beta * (wind_series[-1] - float(np.mean(wind_series))))

    rng = np.random.default_rng(seed)
    paths = np.zeros((path_count, horizon_hours + 1), dtype=float)
    paths[:, 0] = latest
    regime_state = np.full(path_count, initial_regime, dtype=int)
    sigma2 = np.full(path_count, sigma_base**2, dtype=float)
    storm_hours = np.zeros(path_count, dtype=float)
    path_peaks = np.full(path_count, latest, dtype=float)
    garch_alpha = 0.12
    garch_beta = 0.82
    garch_omega = max((1.0 - garch_alpha - garch_beta) * (sigma_base**2), 1e-5)
    wave_upper_cap = max(6.0, float(np.quantile(wave_series, 0.99) * 3.0), latest * 4.0)
    sigma2_cap = max((2.0 * sigma_base) ** 2, 0.35)

    if jump_pool.size == 0:
        jump_pool = rng.normal(loc=jump_mean, scale=jump_std, size=256)

    for step in range(1, horizon_hours + 1):
        u = rng.random(path_count)
        stay_calm = (regime_state == 0) & (u < calm_persistence)
        stay_storm = (regime_state == 1) & (u < storm_persistence)
        regime_state = np.where(stay_calm | stay_storm, regime_state, 1 - regime_state)
        storm_hours += (regime_state == 1).astype(float)

        vol_multiplier = np.where(regime_state == 1, storm_vol_multiplier, 1.0)
        regime_drift = np.where(regime_state == 1, 0.012, -0.004)

        prev = paths[:, step - 1]
        z = rng.standard_t(df=student_t_df, size=path_count) / t_scale
        shocks = np.sqrt(sigma2) * vol_multiplier * z

        jump_occurs = rng.random(path_count) < jump_intensity
        jump_draws = rng.choice(jump_pool, size=path_count, replace=True)
        jump_component = np.where(jump_occurs, jump_draws, 0.0)

        reversion = kappa * (long_run_mean - prev)
        drift_component = drift_trend + wind_force + regime_drift
        nxt = prev + reversion + drift_component + shocks + jump_component
        paths[:, step] = np.clip(nxt, 0.0, wave_upper_cap)
        path_peaks = np.maximum(path_peaks, paths[:, step])

        realized_shock = paths[:, step] - prev - reversion - drift_component - jump_component
        sigma2 = garch_omega + (garch_alpha * (realized_shock**2)) + (garch_beta * sigma2)
        sigma2 = np.clip(sigma2, 1e-5, sigma2_cap)

    q10, q50, q90 = np.quantile(paths, [0.10, 0.50, 0.90], axis=0)
    final_states = paths[:, -1]
    threshold_2m = float(np.mean(final_states >= 2.0))
    threshold_3m = float(np.mean(final_states >= 3.0))
    threshold_3m_anytime = float(np.mean(np.any(paths >= 3.0, axis=1)))
    expected_peak = float(np.mean(path_peaks))
    tail_cut = float(np.quantile(final_states, 0.95))
    tail_slice = final_states[final_states >= tail_cut]
    cvar95 = float(np.mean(tail_slice)) if tail_slice.size else float(tail_cut)

    trajectory_quantiles = [
        {
            "hour": float(hour),
            "p10_m": round(float(q10[hour]), 3),
            "p50_m": round(float(q50[hour]), 3),
            "p90_m": round(float(q90[hour]), 3),
        }
        for hour in range(horizon_hours + 1)
    ]

    sample_count = min(12, path_count)
    sample_indices = rng.choice(path_count, size=sample_count, replace=False)
    sample_paths = np.round(paths[sample_indices], 3).tolist()
    final_sample_count = min(1200, path_count)
    final_sample_indices = rng.choice(path_count, size=final_sample_count, replace=False)
    final_state_sample = np.round(final_states[final_sample_indices], 3).tolist()

    return {
        "status": "ok",
        "model_name": "ou_jump_garch_regime_t",
        "horizon_hours": horizon_hours,
        "path_count": path_count,
        "latest_wave_height_m": round(latest, 3),
        "drift_per_hour": round(float(drift_trend + wind_force), 4),
        "volatility_per_sqrt_hour": round(float(math.sqrt(np.mean(sigma2))), 4),
        "mean_reversion_kappa": round(float(kappa), 4),
        "long_run_mean_m": round(float(long_run_mean), 3),
        "wind_sensitivity_beta": round(float(wind_beta), 4),
        "wind_forcing_component": round(float(wind_force), 4),
        "jump_intensity_per_hour": round(float(jump_intensity), 4),
        "student_t_df": round(float(student_t_df), 2),
        "calm_regime_persistence": round(float(calm_persistence), 4),
        "storm_regime_persistence": round(float(storm_persistence), 4),
        "storm_vol_multiplier": round(float(storm_vol_multiplier), 3),
        "final_p10_m": round(float(np.quantile(final_states, 0.10)), 3),
        "final_p50_m": round(float(np.quantile(final_states, 0.50)), 3),
        "final_p90_m": round(float(np.quantile(final_states, 0.90)), 3),
        "expected_peak_m": round(expected_peak, 3),
        "cvar95_final_m": round(cvar95, 3),
        "probability_exceed_2m": round(threshold_2m, 4),
        "probability_exceed_3m": round(threshold_3m, 4),
        "probability_reach_3m_anytime": round(threshold_3m_anytime, 4),
        "trajectory_quantiles": trajectory_quantiles,
        "sample_paths": sample_paths,
        "final_state_sample": final_state_sample,
        "limitations": [
            "Model uses OU mean reversion + stochastic volatility + jump diffusion + regime switching assumptions.",
            "No external wave model assimilation (e.g., SWAN/WAVEWATCH III) or bathymetry forcing is included.",
        ],
    }


def _hourly_wave_and_wind_series(buoy_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    if buoy_frame.empty or "timestamp" not in buoy_frame.columns or "wave_height_m" not in buoy_frame.columns:
        return np.array([], dtype=float), None

    frame = buoy_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return np.array([], dtype=float), None

    keep_cols = ["wave_height_m"]
    if "wind_speed_mps" in frame.columns:
        keep_cols.append("wind_speed_mps")
    frame = frame[["timestamp"] + keep_cols].set_index("timestamp")
    hourly = frame.resample("1h").mean().interpolate(method="time", limit_direction="both")
    hourly = hourly.dropna(subset=["wave_height_m"])
    if hourly.empty:
        return np.array([], dtype=float), None

    wave = pd.to_numeric(hourly["wave_height_m"], errors="coerce").dropna().to_numpy(dtype=float)
    if wave.size == 0:
        return np.array([], dtype=float), None

    wind = None
    if "wind_speed_mps" in hourly.columns:
        aligned = hourly.loc[hourly["wave_height_m"].notna(), "wind_speed_mps"]
        aligned = pd.to_numeric(aligned, errors="coerce").ffill().bfill()
        if len(aligned) == len(wave):
            wind = aligned.to_numpy(dtype=float)
    return wave, wind


def _estimate_ou_parameters(wave_series: np.ndarray) -> tuple[float, float, np.ndarray]:
    if wave_series.size < 3:
        mu = float(np.mean(wave_series)) if wave_series.size else 0.0
        return mu, 0.05, np.zeros(max(wave_series.size - 1, 1), dtype=float)

    prev = wave_series[:-1]
    nxt = wave_series[1:]
    x = np.column_stack([np.ones_like(prev), prev])
    intercept, beta = np.linalg.lstsq(x, nxt, rcond=None)[0]
    beta = float(np.clip(beta, -0.999, 0.999))

    if 0.02 < beta < 0.999:
        kappa = float(np.clip(-math.log(beta), 0.01, 0.35))
        mu = float(intercept / max(1.0 - beta, 1e-6))
    else:
        mu = float(np.mean(wave_series))
        kappa = 0.06

    fitted = intercept + beta * prev
    residuals = nxt - fitted
    if residuals.size == 0:
        residuals = np.array([0.0], dtype=float)
    return mu, kappa, residuals.astype(float)


def _estimate_student_t_df(residuals: np.ndarray) -> float:
    if residuals.size < 6:
        return 8.0
    kurtosis = float(stats.kurtosis(residuals, fisher=False, bias=False))
    if not np.isfinite(kurtosis) or kurtosis <= 3.05:
        return 8.0
    nu = (4.0 * kurtosis - 6.0) / max(kurtosis - 3.0, 1e-6)
    return float(np.clip(nu, 4.2, 40.0))


def _estimate_regime_parameters(
    wave_diffs: np.ndarray,
    sigma_base: float,
) -> tuple[float, float, float, int]:
    if wave_diffs.size < 12:
        return 0.90, 0.78, 1.6, 0

    abs_diff = np.abs(wave_diffs)
    threshold = float(np.quantile(abs_diff, 0.70))
    regimes = (abs_diff >= threshold).astype(int)
    initial_regime = int(regimes[-1])

    prev = regimes[:-1]
    nxt = regimes[1:]
    count_00 = int(np.sum((prev == 0) & (nxt == 0)))
    count_01 = int(np.sum((prev == 0) & (nxt == 1)))
    count_11 = int(np.sum((prev == 1) & (nxt == 1)))
    count_10 = int(np.sum((prev == 1) & (nxt == 0)))

    p00 = count_00 / max(count_00 + count_01, 1)
    p11 = count_11 / max(count_11 + count_10, 1)
    p00 = float(np.clip(p00, 0.55, 0.98))
    p11 = float(np.clip(p11, 0.50, 0.98))

    calm_vol = float(np.mean(abs_diff[regimes == 0])) if np.any(regimes == 0) else sigma_base
    storm_vol = float(np.mean(abs_diff[regimes == 1])) if np.any(regimes == 1) else max(1.6 * calm_vol, sigma_base)
    storm_multiplier = float(np.clip(storm_vol / max(calm_vol, 1e-4), 1.2, 2.2))
    return p00, p11, storm_multiplier, initial_regime

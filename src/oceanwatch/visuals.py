from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_visuals(
    buoy_frame: pd.DataFrame,
    water_level_frame: pd.DataFrame,
    tide_prediction_frame: pd.DataFrame,
    advanced_analytics: dict[str, Any] | None = None,
) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {}

    if not buoy_frame.empty:
        figures["buoy_conditions"] = _buoy_conditions_figure(buoy_frame)

    if not water_level_frame.empty or not tide_prediction_frame.empty:
        figures["water_level_tides"] = _water_level_figure(water_level_frame, tide_prediction_frame)

    if advanced_analytics:
        figures["risk_distribution"] = _distribution_figure(advanced_analytics)
        figures["scenario_outlooks"] = _scenario_figure(advanced_analytics)

    return figures


def visual_summary(figures: dict[str, go.Figure]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, fig in figures.items():
        summary[name] = {
            "trace_count": len(fig.data),
            "title": fig.layout.title.text if fig.layout and fig.layout.title else name,
        }
    return summary


def _buoy_conditions_figure(frame: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "wave_height_m" in frame.columns:
        wave = frame.dropna(subset=["wave_height_m"])
        fig.add_trace(
            go.Scatter(
                x=wave["timestamp"],
                y=wave["wave_height_m"],
                mode="lines+markers",
                name="Wave Height (m)",
                line={"color": "#6ad4ff", "width": 2.5},
                marker={"size": 6},
                hovertemplate=(
                    "Time: %{x}<br>Wave: %{y:.2f} m<br>"
                    "Meaning: <b>%{text}</b><extra></extra>"
                ),
                text=[_wave_callout(float(value)) for value in wave["wave_height_m"]],
            ),
            secondary_y=False,
        )

    if "wind_speed_mps" in frame.columns:
        wind = frame.dropna(subset=["wind_speed_mps"])
        fig.add_trace(
            go.Scatter(
                x=wind["timestamp"],
                y=wind["wind_speed_mps"],
                mode="lines+markers",
                name="Wind Speed (m/s)",
                line={"color": "#ffd166", "width": 2},
                marker={"size": 5},
                hovertemplate=(
                    "Time: %{x}<br>Wind: %{y:.2f} m/s<br>"
                    "Meaning: <b>%{text}</b><extra></extra>"
                ),
                text=[_wind_callout(float(value)) for value in wind["wind_speed_mps"]],
            ),
            secondary_y=True,
        )

    if "water_temp_c" in frame.columns:
        temp = frame.dropna(subset=["water_temp_c"])
        fig.add_trace(
            go.Scatter(
                x=temp["timestamp"],
                y=temp["water_temp_c"],
                mode="lines",
                name="Water Temp (C)",
                line={"color": "#2ec4b6", "dash": "dot", "width": 2},
                hovertemplate=(
                    "Time: %{x}<br>Water Temp: %{y:.2f} C<br>"
                    "Meaning: <b>%{text}</b><extra></extra>"
                ),
                text=[_temp_callout(float(value)) for value in temp["water_temp_c"]],
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Buoy Dynamics: Wave, Wind, Temperature",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Wave Height (m)", secondary_y=False)
    fig.update_yaxes(title_text="Wind Speed / Water Temp", secondary_y=True)
    return fig


def _water_level_figure(water_level: pd.DataFrame, predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not water_level.empty:
        fig.add_trace(
            go.Scatter(
                x=water_level["timestamp"],
                y=water_level["water_level_m"],
                mode="lines+markers",
                name="Observed Water Level (m)",
                line={"color": "#90e0ef", "width": 2},
                marker={"size": 4},
                hovertemplate=(
                    "Time: %{x}<br>Observed Water Level: %{y:.3f} m<br>"
                    "Meaning: <b>%{text}</b><extra></extra>"
                ),
                text=[_level_callout(float(value)) for value in water_level["water_level_m"]],
            )
        )

    if not predictions.empty:
        color_map = {"H": "#ff6b6b", "L": "#4cc9f0"}
        for tide_type, group in predictions.groupby("tide_type"):
            fig.add_trace(
                go.Scatter(
                    x=group["timestamp"],
                    y=group["predicted_tide_ft"],
                    mode="markers+lines",
                    name=f"Predicted Tide ({tide_type}) ft",
                    marker={"size": 8, "color": color_map.get(str(tide_type), "#7f7f7f")},
                    hovertemplate=(
                        "Time: %{x}<br>Predicted Tide: %{y:.2f} ft<br>"
                        "Meaning: <b>%{text}</b><extra></extra>"
                    ),
                    text=[_tide_callout(str(tide_type), float(value)) for value in group["predicted_tide_ft"]],
                )
            )

    fig.update_layout(
        title="Water Level and Tide Predictions",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        yaxis_title="Water level (m) / Predicted tide (ft)",
        hovermode="x unified",
    )
    return fig


def _distribution_figure(advanced_analytics: dict[str, Any]) -> go.Figure:
    rows = advanced_analytics.get("distribution_profiles", []) or []
    fig = go.Figure()
    if not rows:
        fig.update_layout(title="Distribution Bands (not enough data)")
        return fig

    labels = [row.get("label", row.get("field", "Series")) for row in rows]
    p25 = [row.get("p25", 0) for row in rows]
    p50 = [row.get("p50", 0) for row in rows]
    p75 = [row.get("p75", 0) for row in rows]

    fig.add_trace(go.Bar(x=labels, y=p25, name="P25", marker_color="#6baed6"))
    fig.add_trace(go.Bar(x=labels, y=p50, name="P50", marker_color="#4292c6"))
    fig.add_trace(go.Bar(x=labels, y=p75, name="P75", marker_color="#2171b5"))
    fig.update_layout(
        title="Statistical Distribution Bands (P25/P50/P75)",
        template="plotly_white",
        barmode="group",
        margin={"l": 20, "r": 20, "t": 60, "b": 30},
        hovermode="x unified",
    )
    return fig


def _scenario_figure(advanced_analytics: dict[str, Any]) -> go.Figure:
    outcomes = advanced_analytics.get("simulation_outcomes", []) or []
    fig = go.Figure()
    if not outcomes:
        fig.update_layout(title="Scenario Outlooks (not available)")
        return fig

    labels = [row.get("scenario", "Scenario") for row in outcomes]
    severity = [row.get("severity_score_0_100", 0) for row in outcomes]
    likelihood = [float(row.get("likelihood_probability", 0.0)) * 100 for row in outcomes]
    tooltip = [row.get("interpretation", "") for row in outcomes]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=severity,
            name="Severity Score",
            marker_color="#ff9f1c",
            hovertemplate="Scenario: %{x}<br>Severity: %{y}/100<br>%{text}<extra></extra>",
            text=tooltip,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=likelihood,
            mode="lines+markers",
            name="Likelihood (%)",
            marker={"size": 9, "color": "#2ec4b6"},
            yaxis="y2",
            hovertemplate="Scenario: %{x}<br>Likelihood: %{y:.1f}%<br>%{text}<extra></extra>",
            text=tooltip,
        )
    )
    fig.update_layout(
        title="Scenario Simulation: Severity vs Likelihood",
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 30},
        yaxis={"title": "Severity (0-100)"},
        yaxis2={"title": "Likelihood (%)", "overlaying": "y", "side": "right"},
    )
    return fig


def _wave_callout(value: float) -> str:
    if value >= 2.5:
        return "High surf stress; shoreline risk elevated."
    if value >= 1.2:
        return "Moderate wave energy; monitor local changes."
    return "Low wave energy; calmer marine state."


def _wind_callout(value: float) -> str:
    if value >= 12:
        return "Strong winds can accelerate hazardous sea-state shifts."
    if value >= 7:
        return "Moderate winds may push short-term wave growth."
    return "Light winds; lower near-term forcing."


def _temp_callout(value: float) -> str:
    if value >= 20:
        return "Warm water regime; biological patterns can shift."
    if value <= 8:
        return "Cold-water regime; thermal shock risk can increase."
    return "Typical local thermal range."


def _level_callout(value: float) -> str:
    if value >= 1.0:
        return "Elevated coastal water level; flooding sensitivity higher."
    if value <= -0.2:
        return "Lower than usual level; reduced inundation pressure."
    return "Near baseline coastal level."


def _tide_callout(tide_type: str, value: float) -> str:
    if tide_type == "H":
        return f"High tide peak near {value:.2f} ft can amplify surge sensitivity."
    return f"Low tide trough near {value:.2f} ft can temporarily relieve pressure."

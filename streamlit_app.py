from __future__ import annotations

from datetime import datetime, timezone
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oceanwatch.config import load_settings
from oceanwatch.schemas import AnalysisRequest, RunResult, WaveMonteCarloReport
from oceanwatch.service import OceanWatchService
from oceanwatch.stations import STATION_CATALOG, get_station_by_label


settings = load_settings()

PORT_CATALOG = [
    {"name": "Port of Los Angeles", "coast": "West Coast", "lat": 33.736, "lon": -118.264, "cargo_index": 100, "segment": "Container"},
    {"name": "Port of Long Beach", "coast": "West Coast", "lat": 33.754, "lon": -118.216, "cargo_index": 96, "segment": "Container"},
    {"name": "Port of Oakland", "coast": "West Coast", "lat": 37.798, "lon": -122.281, "cargo_index": 72, "segment": "Intermodal"},
    {"name": "Port of Seattle", "coast": "West Coast", "lat": 47.602, "lon": -122.339, "cargo_index": 68, "segment": "Container"},
    {"name": "Port of Houston", "coast": "Gulf", "lat": 29.728, "lon": -95.251, "cargo_index": 92, "segment": "Energy"},
    {"name": "Port of New Orleans", "coast": "Gulf", "lat": 29.944, "lon": -90.065, "cargo_index": 74, "segment": "Bulk"},
    {"name": "Port of Mobile", "coast": "Gulf", "lat": 30.695, "lon": -88.038, "cargo_index": 58, "segment": "Bulk"},
    {"name": "Port of New York / NJ", "coast": "East Coast", "lat": 40.669, "lon": -74.041, "cargo_index": 94, "segment": "Container"},
    {"name": "Port of Savannah", "coast": "East Coast", "lat": 32.128, "lon": -81.138, "cargo_index": 82, "segment": "Container"},
    {"name": "Port of Norfolk", "coast": "Atlantic", "lat": 36.916, "lon": -76.327, "cargo_index": 76, "segment": "Defense + Container"},
    {"name": "Port of Charleston", "coast": "Atlantic", "lat": 32.781, "lon": -79.927, "cargo_index": 70, "segment": "Container"},
    {"name": "Port of Honolulu", "coast": "Pacific", "lat": 21.306, "lon": -157.864, "cargo_index": 52, "segment": "Island Logistics"},
]


@st.cache_resource
def get_service() -> OceanWatchService:
    return OceanWatchService(settings=settings)


def apply_ocean_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap');
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 12%, rgba(70, 160, 228, 0.30), transparent 32%),
                radial-gradient(circle at 92% 14%, rgba(30, 220, 202, 0.22), transparent 36%),
                linear-gradient(180deg, #021427 0%, #052a45 52%, #0a3b5e 100%);
            color: #ecf7ff;
            font-family: 'Manrope', sans-serif;
        }
        [data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #032339 0%, #0d466c 100%);
            border-right: 1px solid rgba(120, 204, 255, 0.20);
        }
        /* Force bright white sidebar text for headers/labels/help text */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] .st-emotion-cache-10trblm {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ecf7ff;
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.2px;
        }
        p, li, label { color: #ecf7ff; }
        .stMarkdown, .stText, .stCaption { color: #ecf7ff; }
        /* Keep form controls readable on their default light backgrounds */
        [data-testid="stSidebar"] [data-baseweb="select"] * { color: #0f2233 !important; }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #f7fbff !important;
            border-color: rgba(9, 47, 72, 0.45) !important;
            color: #0f2233 !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="select"] div {
            color: #0f2233 !important;
            -webkit-text-fill-color: #0f2233 !important;
        }
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] input {
            color: #0f2233 !important;
            -webkit-text-fill-color: #0f2233 !important;
            background: #f7fbff !important;
        }
        [data-testid="stSidebar"] textarea::placeholder,
        [data-testid="stSidebar"] input::placeholder {
            color: #5a6b7a !important;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"],
        [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"] {
            color: #ffffff !important;
        }
        /* Lock sidebar open: remove collapse/expand controls entirely */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        button[title="Collapse sidebar"],
        button[title="Expand sidebar"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }
        /* Sidebar collapse/expand (double-arrow) visibility fallback */
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        div[data-testid="collapsedControl"] > button,
        div[data-testid="stSidebarCollapsedControl"] > button,
        button[title="Collapse sidebar"],
        button[title="Expand sidebar"] {
            color: #ffffff !important;
            background: rgba(3, 8, 20, 0.80) !important;
            border: 1px solid rgba(166, 230, 255, 0.45) !important;
            border-radius: 10px !important;
            opacity: 1 !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08) inset !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {
            opacity: 1 !important;
            z-index: 9999 !important;
        }
        [data-testid="stSidebarCollapseButton"] button svg,
        [data-testid="collapsedControl"] button svg,
        [data-testid="stSidebarCollapsedControl"] button svg,
        [data-testid="stSidebarCollapseButton"] button span,
        [data-testid="collapsedControl"] button span,
        [data-testid="stSidebarCollapsedControl"] button span,
        [data-testid="stSidebarCollapseButton"] button i,
        [data-testid="collapsedControl"] button i,
        [data-testid="stSidebarCollapsedControl"] button i,
        [data-testid="stSidebarCollapseButton"] button *,
        [data-testid="collapsedControl"] button *,
        [data-testid="stSidebarCollapsedControl"] button *,
        button[title="Collapse sidebar"] svg,
        button[title="Expand sidebar"] svg,
        button[title="Collapse sidebar"] *,
        button[title="Expand sidebar"] *,
        [data-testid="collapsedControl"] *,
        [data-testid="stSidebarCollapsedControl"] * {
            fill: #ffffff !important;
            stroke: #ffffff !important;
            color: #ffffff !important;
            opacity: 1 !important;
            -webkit-text-fill-color: #ffffff !important;
            text-shadow: 0 0 1px rgba(255, 255, 255, 0.25) !important;
        }
        [data-testid="collapsedControl"] svg path,
        [data-testid="stSidebarCollapsedControl"] svg path,
        [data-testid="collapsedControl"] svg g,
        [data-testid="stSidebarCollapsedControl"] svg g {
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }
        [data-testid="stAlertContainer"] > div { border-radius: 12px; }
        [data-testid="stChatMessage"] {
            background: rgba(1, 3, 8, 0.78);
            border: 1px solid rgba(117, 206, 255, 0.30);
            border-radius: 12px;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] span {
            color: #f2f8ff !important;
        }
        .ocean-hero {
            background: linear-gradient(120deg, rgba(7, 113, 175, 0.55), rgba(16, 189, 176, 0.35));
            border: 1px solid rgba(164, 229, 255, 0.34);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.30rem 0.75rem;
            font-size: 0.84rem;
            margin-right: 0.35rem;
            border: 1px solid rgba(255,255,255,0.26);
            margin-bottom: 0.3rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }
        .metric-card {
            background: rgba(12, 72, 105, 0.60);
            border: 1px solid rgba(121, 214, 255, 0.28);
            border-radius: 14px;
            padding: 0.75rem 0.8rem;
            min-height: 112px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.16);
        }
        .metric-label {
            font-size: 0.88rem;
            opacity: 0.9;
            margin-bottom: 0.18rem;
            line-height: 1.15;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.04;
            color: #f7fdff;
        }
        .metric-sub {
            margin-top: 0.2rem;
            font-size: 0.80rem;
            opacity: 0.8;
            line-height: 1.15;
        }
        .insight-strip {
            background: rgba(9, 97, 153, 0.35);
            border: 1px solid rgba(150, 224, 255, 0.28);
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
            margin: 0.4rem 0 1rem 0;
        }
        .workflow-card {
            background: rgba(9, 53, 88, 0.52);
            border: 1px solid rgba(146, 224, 255, 0.30);
            border-radius: 12px;
            padding: 0.6rem 0.7rem;
            margin-bottom: 0.45rem;
        }
        .workflow-title {
            font-size: 1rem;
            font-weight: 700;
        }
        .workflow-sub {
            font-size: 0.86rem;
            opacity: 0.9;
            margin-top: 0.12rem;
        }
        [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: rgba(6, 36, 62, 0.52);
            border: 1px solid rgba(146, 224, 255, 0.28);
            border-radius: 12px;
            padding: 0.3rem;
        }
        button[data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(120deg, rgba(15, 121, 190, 0.85), rgba(38, 200, 187, 0.58));
            color: #f8fdff;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(130, 207, 245, 0.22);
            border-radius: 12px;
            overflow: hidden;
        }
        .kpi-ribbon {
            background: linear-gradient(100deg, rgba(7, 84, 130, 0.55), rgba(23, 167, 165, 0.42));
            border: 1px solid rgba(160, 229, 255, 0.30);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.8rem;
        }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(180deg, #03050d 0%, #0a0f1f 100%) !important;
            color: #f4fbff !important;
            border: 1px solid rgba(145, 221, 255, 0.35) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: rgba(140, 228, 255, 0.65) !important;
            box-shadow: 0 0 0 1px rgba(100, 200, 255, 0.22) inset;
        }
        [data-testid="stPlotlyChart"] > div {
            background: #040710;
            border: 1px solid rgba(118, 208, 255, 0.20);
            border-radius: 12px;
            padding: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_station_map(selected_station_key: str, result: RunResult | None = None) -> None:
    severity = 0
    if result is not None:
        severity = int(result.metrics.advanced_analytics.get("severity", {}).get("score_0_100", 0))

    rows = []
    for station in STATION_CATALOG:
        rows.append(
            {
                "name": station.display_name,
                "coast": station.coast,
                "lat": station.latitude,
                "lon": station.longitude,
                "key": station.key,
                "ndbc": station.ndbc_station_id,
                "coops": station.coops_station_id,
                "status": station.validation_status,
            }
        )
    frame = pd.DataFrame(rows)

    fig = go.Figure()
    coast_colors = {
        "West Coast": "#6ad4ff",
        "East Coast": "#9ae17c",
        "Gulf": "#ffd166",
        "Pacific": "#7a8cff",
        "Atlantic": "#ff9a76",
    }
    for coast_name, group in frame.groupby("coast"):
        fig.add_trace(
            go.Scattergeo(
                lon=group["lon"],
                lat=group["lat"],
                text=group["name"],
                mode="markers",
                marker={
                    "size": 9,
                    "color": coast_colors.get(coast_name, "#8ecae6"),
                    "line": {"color": "white", "width": 1},
                },
                name=coast_name,
                customdata=group[["ndbc", "coops", "status"]],
                hovertemplate=(
                    "<b>%{text}</b><br>Coast: "
                    + coast_name
                    + "<br>NDBC: %{customdata[0]}<br>CO-OPS: %{customdata[1]}"
                    + "<br>Validation: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    selected = frame[frame["key"] == selected_station_key]
    if not selected.empty:
        fig.add_trace(
            go.Scattergeo(
                lon=selected["lon"],
                lat=selected["lat"],
                text=selected["name"],
                mode="markers",
                marker={
                    "size": 17,
                    "symbol": "star",
                    "color": "#ffdd57",
                    "line": {"color": "#06263e", "width": 2},
                },
                name="Selected Station",
                customdata=[[severity]],
                hovertemplate=(
                    "<b>%{text}</b><br>Selected for analysis"
                    "<br>Current Severity Score: %{customdata[0]}/100"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="North America Coastal Station Network",
        title_font={"color": "#eaf6ff", "size": 27},
        height=340,
        margin={"l": 0, "r": 0, "t": 38, "b": 0},
        paper_bgcolor="#040710",
        plot_bgcolor="#040710",
        legend={"orientation": "h", "y": 1.06, "x": 0},
        legend_font={"color": "#dff3ff"},
        font={"color": "#eaf6ff"},
    )
    fig.update_geos(
        scope="north america",
        projection_type="mercator",
        showland=True,
        landcolor="#143d58",
        showcountries=True,
        countrycolor="#5a9abf",
        showocean=True,
        oceancolor="#05263f",
        lataxis={"range": [15, 65]},
        lonaxis={"range": [-170, -55]},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover stations for NOAA IDs and coast context. The selected star uses current severity score.")


def render_source_health(health: dict[str, str]) -> None:
    st.subheader("Source Health")
    cols = st.columns(3)
    labels = ["ndbc_buoy", "coops_water_level", "coops_tide_predictions"]
    for col, label in zip(cols, labels):
        status = health.get(label, "missing")
        color = "#13d17a" if status == "ok" else "#ffb703" if status in {"empty", "missing"} else "#ff6b6b"
        col.markdown(
            (
                f"<div class='status-pill' style='background: {color}22; border-color: {color}66;'>"
                f"{label.replace('_', ' ').title()}: {status}</div>"
            ),
            unsafe_allow_html=True,
        )


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _source_freshness_minutes(result: RunResult) -> float | None:
    ages: list[float] = []
    now = datetime.now(timezone.utc)
    for payload in result.source_payloads.values():
        if not payload.available:
            continue
        fetched = payload.fetched_at_utc
        if isinstance(fetched, str):
            parsed = pd.to_datetime(fetched, utc=True, errors="coerce")
            if pd.isna(parsed):
                continue
            fetched_dt = parsed.to_pydatetime()
        else:
            fetched_dt = fetched
        if fetched_dt.tzinfo is None:
            fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
        ages.append((now - fetched_dt).total_seconds() / 60.0)
    if not ages:
        return None
    return max(ages)


def _anomaly_density_pct(result: RunResult) -> float | None:
    metrics = result.metrics.buoy_metrics + result.metrics.water_level_metrics
    total_count = sum(int(item.count) for item in metrics)
    if total_count <= 0:
        return None
    anomaly_count = sum(int(item.anomaly_count) for item in metrics)
    return (anomaly_count / total_count) * 100.0


def _max_regression_exceedance_pct(result: RunResult) -> float | None:
    regressions = result.metrics.advanced_analytics.get("regression_signals", [])
    probs = [_as_float(item.get("exceedance_probability")) for item in regressions if isinstance(item, dict)]
    probs = [item for item in probs if item is not None]
    if not probs:
        return None
    return max(probs) * 100.0


def _format_card_value(value: object, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{suffix}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}{suffix}"
    return str(value)


def render_metric_cards(result: RunResult, mc_report: WaveMonteCarloReport | None = None) -> None:
    st.subheader("Metric Cards")
    metric_cards = result.metrics.metric_cards
    source_ok = sum(1 for value in result.health.values() if value == "ok")
    total_rows = int(sum(payload.row_count for payload in result.source_payloads.values()))
    freshness_mins = _source_freshness_minutes(result)
    anomaly_density = _anomaly_density_pct(result)
    max_exceedance = _max_regression_exceedance_pct(result)
    adk_badge = result.adk_status.replace("_", " ").title()
    severity = result.metrics.advanced_analytics.get("severity", {})
    severity_level = str(severity.get("level", "Unknown"))

    def _build_card(label: str, value: object, unit: str = "", decimals: int = 2, help_text: str = "") -> dict[str, str]:
        suffix = unit
        text = _format_card_value(value, decimals=decimals, suffix=suffix)
        return {"label": label, "value": text, "help": help_text}

    cards = [
        _build_card("Wave Height", metric_cards.get("latest_wave_height_m"), " m", 2, "Latest buoy observation"),
        _build_card("Wind Speed", metric_cards.get("latest_wind_speed_mps"), " m/s", 2, "Latest sustained wind"),
        _build_card("Water Temp", metric_cards.get("latest_water_temp_c"), " C", 2, "Latest surface water"),
        _build_card("Water Level", metric_cards.get("latest_water_level_m"), " m", 2, "Latest observed level"),
        _build_card("Tide Range", metric_cards.get("predicted_tide_range_ft"), " ft", 2, "Predicted high-low range"),
        _build_card("Severity Score", metric_cards.get("severity_score_0_100"), "", 0, f"Severity level: {severity_level}"),
        _build_card("Data Coverage", total_rows, " rows", 0, f"Rows across {source_ok}/3 healthy sources"),
        _build_card("Source Freshness", freshness_mins, " min", 1, "Oldest feed age in minutes"),
        _build_card("Anomaly Density", anomaly_density, "%", 1, "Anomalies per usable observation"),
        _build_card("Model Exceed Risk", max_exceedance, "%", 1, "Max 6h exceedance probability"),
        _build_card("ADK Status", adk_badge, "", 0, "Framework status"),
        _build_card("Runtime", result.runtime_seconds, " s", 2, "End-to-end analysis runtime"),
    ]

    if mc_report is not None:
        cards.extend(
            [
                _build_card("MC P(end>2m)", mc_report.probability_exceed_2m * 100.0, "%", 1, "Monte Carlo exceedance risk"),
                _build_card("MC P(any>3m)", mc_report.probability_reach_3m_anytime * 100.0, "%", 1, "Tail-path touch probability"),
                _build_card("MC Tail Risk", mc_report.cvar95_final_m, " m", 2, "CVaR95 final-state wave"),
                _build_card("MC Expected Peak", mc_report.expected_peak_m, " m", 2, "Expected max path height"),
                _build_card("MC Drift", mc_report.drift_per_hour, " /h", 4, "Estimated drift component"),
                _build_card("MC ADK Gate", mc_report.mc_adk_status.replace("_", " ").title(), "", 0, "Interpretation gate status"),
            ]
        )

    row_width = 4
    for start in range(0, len(cards), row_width):
        row_cards = cards[start : start + row_width]
        cols = st.columns(row_width)
        for idx, card in enumerate(row_cards):
            with cols[idx]:
                st.metric(card["label"], card["value"], help=card["help"])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(r * c)


def _coast_similarity(port_coast: str, station_coast: str) -> float:
    if port_coast == station_coast:
        return 1.0
    if {port_coast, station_coast} <= {"Atlantic", "East Coast"}:
        return 0.82
    if {port_coast, station_coast} <= {"Pacific", "West Coast"}:
        return 0.82
    return 0.58


def build_port_impact_frame(station: dict[str, object], result: RunResult) -> pd.DataFrame:
    severity_score = float(result.metrics.advanced_analytics.get("severity", {}).get("score_0_100", 0.0))
    exceedance = _max_regression_exceedance_pct(result) or 0.0
    anomaly_density = _anomaly_density_pct(result) or 0.0
    station_lat = float(station.get("latitude", 0.0))
    station_lon = float(station.get("longitude", 0.0))
    station_coast = str(station.get("coast", "West Coast"))
    rows = []
    for port in PORT_CATALOG:
        distance_km = _haversine_km(station_lat, station_lon, float(port["lat"]), float(port["lon"]))
        distance_weight = float(np.exp(-distance_km / 2200.0))
        coast_weight = _coast_similarity(str(port["coast"]), station_coast)
        cargo_norm = float(port["cargo_index"]) / 100.0
        dynamic_risk = (
            (0.50 * (severity_score / 100.0))
            + (0.28 * (exceedance / 100.0))
            + (0.22 * (anomaly_density / 100.0))
        )
        risk_index = 100.0 * ((0.65 * dynamic_risk * distance_weight * coast_weight) + (0.35 * cargo_norm))
        risk_index = float(np.clip(risk_index, 1.0, 99.0))
        risk_band = "High" if risk_index >= 70 else "Moderate" if risk_index >= 45 else "Low"
        rows.append(
            {
                "port": port["name"],
                "coast": port["coast"],
                "segment": port["segment"],
                "cargo_index": int(port["cargo_index"]),
                "distance_km": round(distance_km, 1),
                "risk_index": round(risk_index, 1),
                "risk_band": risk_band,
                "lat": float(port["lat"]),
                "lon": float(port["lon"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("risk_index", ascending=False).reset_index(drop=True)
    return frame


def render_ocean_lens(station: dict[str, object], result: RunResult) -> None:
    st.subheader("Ocean Lens: Port Impact Intelligence")
    st.caption("Resilient map + analytics for likely U.S. port stress linked to selected station conditions.")
    port_frame = build_port_impact_frame(station, result)
    top_row = st.columns([1.45, 1], gap="large")
    with top_row[0]:
        fig = go.Figure()
        fig.add_trace(
            go.Scattergeo(
                lon=port_frame["lon"],
                lat=port_frame["lat"],
                text=port_frame["port"],
                mode="markers",
                marker={
                    "size": (port_frame["cargo_index"] / 3.0) + 8,
                    "color": port_frame["risk_index"],
                    "colorscale": "Turbo",
                    "cmin": 0,
                    "cmax": 100,
                    "line": {"width": 1, "color": "white"},
                    "colorbar": {"title": "Port Risk"},
                },
                customdata=port_frame[["coast", "segment", "distance_km", "risk_index", "risk_band"]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>Coast: %{customdata[0]}<br>Segment: %{customdata[1]}"
                    "<br>Distance from station: %{customdata[2]} km"
                    "<br>Risk index: %{customdata[3]} (%{customdata[4]})<extra></extra>"
                ),
                name="Ports",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[float(station.get("longitude", 0.0))],
                lat=[float(station.get("latitude", 0.0))],
                text=[str(station.get("display_name", "Selected Station"))],
                mode="markers",
                marker={"size": 18, "symbol": "star", "color": "#ffe66d", "line": {"color": "#08314d", "width": 2}},
                name="Selected Station",
                hovertemplate="<b>%{text}</b><br>Origin station for risk propagation<extra></extra>",
            )
        )
        fig.update_layout(
            title="Port Risk Propagation Map",
            height=420,
            margin={"l": 0, "r": 0, "t": 44, "b": 0},
            paper_bgcolor="#040710",
            plot_bgcolor="#040710",
            font={"color": "#eaf6ff"},
            legend={"orientation": "h", "y": 1.06},
        )
        fig.update_geos(
            scope="north america",
            projection_type="mercator",
            showland=True,
            landcolor="#163f5b",
            showcountries=True,
            countrycolor="#6da8ca",
            showocean=True,
            oceancolor="#062740",
            lataxis={"range": [15, 65]},
            lonaxis={"range": [-170, -55]},
        )
        st.plotly_chart(fig, use_container_width=True)
    with top_row[1]:
        top_ports = port_frame.head(8).iloc[::-1]
        bar = go.Figure()
        bar.add_trace(
            go.Bar(
                y=top_ports["port"],
                x=top_ports["risk_index"],
                orientation="h",
                marker={"color": top_ports["risk_index"], "colorscale": "Turbo", "cmin": 0, "cmax": 100},
                customdata=top_ports[["distance_km", "cargo_index", "risk_band"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>Risk Index: %{x:.1f}<br>Distance: %{customdata[0]} km"
                    "<br>Cargo Index: %{customdata[1]}<br>Band: %{customdata[2]}<extra></extra>"
                ),
                name="Risk",
            )
        )
        bar.update_layout(
            title="Top Port Exposure (Selected Conditions)",
            xaxis_title="Risk Index (0-100)",
            yaxis_title="",
            height=420,
            margin={"l": 0, "r": 10, "t": 44, "b": 10},
            paper_bgcolor="#040710",
            plot_bgcolor="#040710",
            font={"color": "#eaf6ff"},
        )
        st.plotly_chart(bar, use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Highest Port Risk", f"{port_frame['risk_index'].max():.1f}")
    k2.metric("Average Port Risk", f"{port_frame['risk_index'].mean():.1f}")
    k3.metric("Ports in High Band", f"{int((port_frame['risk_band'] == 'High').sum())}")
    k4.metric("Nearest Top-Risk Port", port_frame.iloc[0]["port"])

    st.dataframe(
        port_frame[["port", "coast", "segment", "distance_km", "cargo_index", "risk_index", "risk_band"]],
        use_container_width=True,
        hide_index=True,
    )


def _trend_direction_text(result: RunResult, field: str) -> str:
    metric_map = {m.field: m for m in (result.metrics.buoy_metrics + result.metrics.water_level_metrics)}
    metric = metric_map.get(field)
    if not metric or metric.slope_per_hour is None:
        return "stable"
    slope = float(metric.slope_per_hour)
    if slope > 0.02:
        return "rising"
    if slope < -0.02:
        return "falling"
    return "stable"


def build_tab_hypothesis_synthesis(
    result: RunResult,
    mc_report: WaveMonteCarloReport | None = None,
) -> dict[str, dict[str, object]]:
    advanced = result.metrics.advanced_analytics or {}
    severity = result.metrics.advanced_analytics.get("severity", {})
    severity_score = int(severity.get("score_0_100", 0))
    severity_level = str(severity.get("level", "Unknown"))
    exceed = _max_regression_exceedance_pct(result) or 0.0
    anomaly_density = _anomaly_density_pct(result) or 0.0
    wave_direction = _trend_direction_text(result, "wave_height_m")
    port_frame = build_port_impact_frame(result.station, result)
    top_port = str(port_frame.iloc[0]["port"]) if not port_frame.empty else "top regional ports"
    top_port_risk = float(port_frame.iloc[0]["risk_index"]) if not port_frame.empty else 0.0
    high_port_count = int((port_frame["risk_band"] == "High").sum()) if not port_frame.empty else 0
    avg_port_risk = float(port_frame["risk_index"].mean()) if not port_frame.empty else 0.0
    total_rows = int(sum(payload.row_count for payload in result.source_payloads.values()))
    source_ok = int(sum(1 for status in result.health.values() if status == "ok"))
    freshness = _source_freshness_minutes(result)

    regressions = advanced.get("regression_signals", [])
    top_regression = None
    if regressions:
        top_regression = max(
            regressions,
            key=lambda row: float(row.get("exceedance_probability", 0.0) or 0.0),
        )
    reg_label = str(top_regression.get("label", "N/A")) if isinstance(top_regression, dict) else "N/A"
    reg_prob = (float(top_regression.get("exceedance_probability", 0.0)) * 100.0) if isinstance(top_regression, dict) else 0.0
    reg_slope = float(top_regression.get("slope_per_hour", 0.0) or 0.0) if isinstance(top_regression, dict) else 0.0

    patterns = advanced.get("irregular_patterns", [])
    top_pattern = patterns[0] if patterns else None
    pattern_text = (
        f"{top_pattern.get('label', top_pattern.get('field', 'signal'))} at {top_pattern.get('timestamp_utc', 'n/a')}"
        if isinstance(top_pattern, dict)
        else "No strong irregular spike/drop detected in current window."
    )

    executive_payload: dict[str, object] = {
        "summary": (
            f"Anchor interpretation: this run indicates a {severity_level.lower()} coastal-stress regime "
            f"({severity_score}/100) aligned with a {wave_direction} wave regime and modeled exceedance risk "
            f"up to {exceed:.1f}%."
        ),
        "points": [
            f"Composite severity score = {severity_score}/100 ({severity_level}).",
            f"Highest regression exceedance signal = {reg_label} at {reg_prob:.1f}%.",
            f"Anomaly density = {anomaly_density:.1f}% across buoy + water-level observations.",
            f"Primary irregular pattern evidence: {pattern_text}",
        ],
    }
    geo_payload: dict[str, object] = {
        "summary": (
            "Geo Lens shows how the station-level hypothesis propagates into likely port-operational stress zones."
        ),
        "points": [
            f"Top exposed port is {top_port} with risk index {top_port_risk:.1f}/100.",
            f"Average port risk index across tracked ports = {avg_port_risk:.1f}.",
            f"Ports in High band = {high_port_count}, indicating concentrated operational sensitivity.",
            f"This supports the executive hypothesis by translating ocean-state stress into logistics-impact geography.",
        ],
    }
    quant_payload: dict[str, object] = {
        "summary": (
            "Quant Lab validates the hypothesis with explainable EDA evidence from regressions, anomalies, and scenarios."
        ),
        "points": [
            f"Strongest modeled risk driver = {reg_label} with exceedance probability {reg_prob:.1f}%.",
            f"Associated trend slope = {reg_slope:+.4f} per hour, quantifying direction and speed of change.",
            f"Anomaly density = {anomaly_density:.1f}%, showing how unusual current behavior is versus baseline.",
            f"Irregular-pattern evidence used for hypothesis grounding: {pattern_text}",
        ],
    }

    if mc_report is None:
        mc_payload: dict[str, object] = {
            "summary": (
                "Monte Carlo handoff has not run yet, so the final hypothesis currently reflects observed/regression evidence only."
            ),
            "points": [
                "Run Monte Carlo to test path uncertainty and tail outcomes over your selected horizon.",
                "This adds forward-looking stress scenarios beyond static trend extrapolation.",
            ],
        }
    elif mc_report.mc_hypothesis_ready:
        mc_payload = {
            "summary": (
                "Monte Carlo evidence refines the executive hypothesis by quantifying future path uncertainty and tail exposure."
            ),
            "points": [
                f"P(end > 2m) = {mc_report.probability_exceed_2m*100:.1f}% over the selected horizon.",
                f"P(any path > 3m) = {mc_report.probability_reach_3m_anytime*100:.1f}%.",
                f"Final-state median = {mc_report.final_p50_m:.2f} m; tail-risk CVaR95 = {mc_report.cvar95_final_m:.2f} m.",
                "These probabilities calibrate how aggressively to operationalize the executive hypothesis.",
            ],
        }
    else:
        mc_payload = {
            "summary": (
                "Monte Carlo stats are available, but ADK interpretation is blocked; this limits narrative refinement of the final hypothesis."
            ),
            "points": [
                f"Simulation still produced quantitative signals (P(end>2m)={mc_report.probability_exceed_2m*100:.1f}%).",
                f"Narrative gate status = {mc_report.mc_adk_status.replace('_', ' ')}.",
                "Executive hypothesis remains primarily anchored to observed EDA until MC ADK interpretation succeeds.",
            ],
        }
    data_payload: dict[str, object] = {
        "summary": (
            "Data Room ties each hypothesis claim to reproducible runtime evidence and source-health context."
        ),
        "points": [
            f"Total collected rows = {total_rows} from {source_ok}/3 healthy NOAA sources.",
            f"Source freshness (oldest feed age) = {freshness:.1f} min." if freshness is not None else "Source freshness unavailable in this run.",
            "Exportable CSV tables provide an auditable evidence trail for every tab-level claim.",
            "This is the proof layer backing the executive hypothesis with inspectable data lineage.",
        ],
    }
    return {
        "Executive Brief": executive_payload,
        "Geo Lens": geo_payload,
        "Quant Lab": quant_payload,
        "Monte Carlo Lab": mc_payload,
        "Data Room": data_payload,
    }


def render_tab_hypothesis_synthesis(tab_name: str, synthesis_payload: dict[str, object], executive_hypothesis: str) -> None:
    summary = escape(str(synthesis_payload.get("summary", "")))
    raw_points = synthesis_payload.get("points", [])
    points = []
    if isinstance(raw_points, list):
        for item in raw_points[:4]:
            points.append(escape(str(item)))
    hypothesis_anchor = executive_hypothesis.strip()
    if len(hypothesis_anchor) > 220:
        hypothesis_anchor = hypothesis_anchor[:217].rstrip() + "..."
    anchor_html = escape(hypothesis_anchor)
    points_html = "".join(f"<li>{item}</li>" for item in points)
    st.markdown(
        (
            "<div class='insight-strip'>"
            f"<b>{tab_name} Hypothesis Contribution</b><br>"
            f"<span style='opacity:0.92;'><b>Executive hypothesis anchor:</b> \"{anchor_html}\"</span><br>"
            f"<span style='opacity:0.96;'>{summary}</span>"
            f"<ul style='margin-top:0.35rem; margin-bottom:0;'>{points_html}</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_advanced_panels(result: RunResult) -> None:
    advanced = result.metrics.advanced_analytics
    severity = advanced.get("severity", {})

    st.subheader("Severity and Likelihood")
    score = int(severity.get("score_0_100", 0))
    st.markdown(
        f"<div class='insight-strip'><b>{advanced.get('risk_headline', 'No risk headline available.')}</b></div>",
        unsafe_allow_html=True,
    )
    st.progress(score / 100)
    st.caption(f"Severity score: {score}/100 ({severity.get('level', 'Unknown')})")

    drivers = severity.get("drivers", [])
    if drivers:
        st.markdown("**Main risk drivers**")
        for row in drivers[:4]:
            st.write(f"- {row.get('name')}: {row.get('score')}/100")

    regressions = advanced.get("regression_signals", [])
    if regressions:
        st.markdown("**Regression and exceedance likelihoods**")
        reg_df = pd.DataFrame(regressions)
        if "exceedance_probability" in reg_df.columns:
            reg_df["exceedance_pct"] = pd.to_numeric(reg_df["exceedance_probability"], errors="coerce") * 100.0
        if "projected_6h" in reg_df.columns and "threshold" in reg_df.columns:
            reg_df["gap_vs_threshold"] = pd.to_numeric(reg_df["projected_6h"], errors="coerce") - pd.to_numeric(
                reg_df["threshold"], errors="coerce"
            )
        show_cols = [
            "label",
            "slope_per_hour",
            "r_squared",
            "p_value",
            "projected_6h",
            "threshold",
            "threshold_units",
            "exceedance_pct",
            "gap_vs_threshold",
            "risk_band",
        ]
        existing = [col for col in show_cols if col in reg_df.columns]
        chart_cols = st.columns(2, gap="large")
        with chart_cols[0]:
            slope_fig = go.Figure()
            slope_fig.add_trace(
                go.Scatter(
                    x=reg_df.get("slope_per_hour", pd.Series(dtype=float)),
                    y=reg_df.get("exceedance_pct", pd.Series(dtype=float)),
                    mode="markers+text",
                    text=reg_df.get("label", pd.Series(dtype=str)),
                    textposition="top center",
                    marker={
                        "size": (pd.to_numeric(reg_df.get("r_squared", 0), errors="coerce").fillna(0) * 30) + 14,
                        "color": pd.to_numeric(reg_df.get("exceedance_pct", 0), errors="coerce").fillna(0),
                        "colorscale": "Viridis",
                        "line": {"color": "white", "width": 1},
                        "showscale": True,
                        "colorbar": {"title": "Exceedance %"},
                    },
                    customdata=reg_df.get("risk_band", pd.Series(dtype=str)),
                    hovertemplate=(
                        "<b>%{text}</b><br>Slope/hr: %{x:.4f}<br>Exceedance: %{y:.1f}%"
                        "<br>Band: %{customdata}<extra></extra>"
                    ),
                    name="Regression Signal",
                )
            )
            slope_fig.add_hline(y=50, line_dash="dot", line_color="#ffcc70")
            slope_fig.add_vline(x=0, line_dash="dash", line_color="#90e0ef")
            slope_fig.update_layout(
                title="Trend Slope vs Exceedance Risk",
                xaxis_title="Slope Per Hour",
                yaxis_title="Exceedance Probability (%)",
                height=360,
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
                paper_bgcolor="#040710",
                plot_bgcolor="#040710",
                font={"color": "#eaf6ff"},
            )
            st.plotly_chart(slope_fig, use_container_width=True)
        with chart_cols[1]:
            compare_fig = go.Figure()
            compare_fig.add_trace(
                go.Bar(
                    x=reg_df["label"],
                    y=reg_df.get("projected_6h", 0),
                    name="Projected (6h)",
                    marker_color="#6ad4ff",
                    hovertemplate="<b>%{x}</b><br>Projected: %{y:.3f}<extra></extra>",
                )
            )
            compare_fig.add_trace(
                go.Bar(
                    x=reg_df["label"],
                    y=reg_df.get("threshold", 0),
                    name="Threshold",
                    marker_color="#ffd166",
                    hovertemplate="<b>%{x}</b><br>Threshold: %{y:.3f}<extra></extra>",
                )
            )
            compare_fig.update_layout(
                title="Projected Level vs Threshold",
                barmode="group",
                height=360,
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
                paper_bgcolor="#040710",
                plot_bgcolor="#040710",
                font={"color": "#eaf6ff"},
            )
            st.plotly_chart(compare_fig, use_container_width=True)
        st.dataframe(reg_df[existing], use_container_width=True, hide_index=True)

    patterns = advanced.get("irregular_patterns", [])
    if patterns:
        st.markdown("**Irregular patterns detected**")
        for row in patterns[:6]:
            st.write(
                f"- {row.get('label', row.get('field'))} at {row.get('timestamp_utc')}: "
                f"{row.get('value')} {row.get('units')} ({row.get('interpretation')})"
            )

    render_wave_outlook(advanced)

    simulations = advanced.get("simulation_outcomes", [])
    if simulations:
        st.markdown("**Simulation outcomes (tool-generated)**")
        sim_df = pd.DataFrame(simulations)
        if "likelihood_probability" in sim_df.columns:
            sim_df["likelihood_pct"] = pd.to_numeric(sim_df["likelihood_probability"], errors="coerce") * 100.0
        sim_cols = st.columns(2, gap="large")
        with sim_cols[0]:
            sim_scatter = go.Figure()
            sim_scatter.add_trace(
                go.Scatter(
                    x=sim_df.get("likelihood_pct", pd.Series(dtype=float)),
                    y=sim_df.get("severity_score_0_100", pd.Series(dtype=float)),
                    mode="markers+text",
                    text=sim_df.get("scenario", pd.Series(dtype=str)),
                    textposition="top center",
                    marker={
                        "size": (pd.to_numeric(sim_df.get("projected_wave_height_m", 0), errors="coerce").fillna(0) * 10) + 12,
                        "color": pd.to_numeric(sim_df.get("severity_score_0_100", 0), errors="coerce").fillna(0),
                        "colorscale": "Turbo",
                        "line": {"color": "white", "width": 1},
                    },
                    customdata=sim_df.get("interpretation", pd.Series(dtype=str)),
                    hovertemplate=(
                        "<b>%{text}</b><br>Likelihood: %{x:.1f}%<br>Severity: %{y:.1f}/100"
                        "<br>%{customdata}<extra></extra>"
                    ),
                    name="Scenario",
                )
            )
            sim_scatter.update_layout(
                title="Scenario Risk Matrix",
                xaxis_title="Likelihood (%)",
                yaxis_title="Severity Score (0-100)",
                height=360,
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
                paper_bgcolor="#040710",
                plot_bgcolor="#040710",
                font={"color": "#eaf6ff"},
            )
            st.plotly_chart(sim_scatter, use_container_width=True)
        with sim_cols[1]:
            outlook_fig = go.Figure()
            outlook_fig.add_trace(
                go.Bar(
                    x=sim_df.get("scenario", pd.Series(dtype=str)),
                    y=sim_df.get("projected_wave_height_m", pd.Series(dtype=float)),
                    marker_color="#59c3ff",
                    name="Projected Wave Height (m)",
                    hovertemplate="<b>%{x}</b><br>Wave: %{y:.2f} m<extra></extra>",
                )
            )
            if "projected_water_level_m" in sim_df.columns:
                outlook_fig.add_trace(
                    go.Scatter(
                        x=sim_df.get("scenario", pd.Series(dtype=str)),
                        y=sim_df.get("projected_water_level_m", pd.Series(dtype=float)),
                        mode="lines+markers",
                        line={"color": "#ffd166", "width": 2},
                        marker={"size": 8},
                        name="Projected Water Level (m)",
                        yaxis="y2",
                        hovertemplate="<b>%{x}</b><br>Water level: %{y:.2f} m<extra></extra>",
                    )
                )
            outlook_fig.update_layout(
                title="Projected Ocean State by Scenario",
                height=360,
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
                paper_bgcolor="#040710",
                plot_bgcolor="#040710",
                font={"color": "#eaf6ff"},
                yaxis={"title": "Wave Height (m)"},
                yaxis2={"title": "Water Level (m)", "overlaying": "y", "side": "right"},
            )
            st.plotly_chart(outlook_fig, use_container_width=True)
        st.dataframe(sim_df, use_container_width=True, hide_index=True)


def render_agent_workflow(result: RunResult, mc_report: WaveMonteCarloReport | None = None) -> None:
    st.subheader("What the Agents Are Trying to Do")
    buoy_status = "complete" if result.health.get("ndbc_buoy") == "ok" else "degraded"
    tide_ok = (
        result.health.get("coops_water_level") == "ok"
        and result.health.get("coops_tide_predictions") == "ok"
    )
    tide_status = "complete" if tide_ok else "degraded"
    risk_status = "complete" if result.metrics.advanced_analytics else "in progress"
    synthesis_status = "complete" if result.adk_status in {"success", "recovery_success"} else "fallback mode"
    mc_status = "complete" if mc_report is not None else "available"

    cards = [
        (
            "Coordinator Agent",
            "Interprets your question, picks scope, and routes tasks to specialists.",
            "complete",
        ),
        (
            "Buoy Agent",
            "Pull NDBC wave, wind, and temperature data and compute buoy-side diagnostics.",
            buoy_status,
        ),
        (
            "Tide Agent",
            "Pull CO-OPS water levels and tide predictions, then frame coastal context.",
            tide_status,
        ),
        (
            "Risk Agent",
            "Run regressions, likelihoods, irregular-pattern detection, and scenario scoring.",
            risk_status,
        ),
        (
            "Synthesis Agent",
            "Build one hypothesis grounded in tool-derived evidence and caveats.",
            synthesis_status,
        ),
        (
            "Monte Carlo Handoff Agent",
            "Runs only after your selection of horizon and path count, then returns a simulation-driven hypothesis.",
            mc_status,
        ),
    ]

    colors = {
        "complete": "#2ec4b6",
        "degraded": "#ffd166",
        "in progress": "#8ecae6",
        "fallback mode": "#ff9f1c",
        "available": "#89cff0",
    }
    for title, summary, status in cards:
        color = colors.get(status, "#8ecae6")
        st.markdown(
            (
                "<div class='workflow-card'>"
                f"<div class='workflow-title'>{title} · "
                f"<span style='color:{color}; text-transform:uppercase; font-size:0.8rem;'>{status}</span></div>"
                f"<div class='workflow-sub'>{summary}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    st.caption(
        "This panel shows intent and current mode; it is not a raw backend event trace."
    )


def render_wave_outlook(advanced: dict[str, object]) -> None:
    wave_outlook = advanced.get("wave_outlook", {}) if advanced else {}
    if not wave_outlook:
        return
    st.markdown("**Wave outlook (next 24h envelope)**")
    st.info(
        f"Trend: {wave_outlook.get('trend_label', 'unknown')} | "
        f"Expected 24h band: {wave_outlook.get('estimated_24h_low_m', 'n/a')} m "
        f"to {wave_outlook.get('estimated_24h_high_m', 'n/a')} m"
    )
    for row in wave_outlook.get("guidance", [])[:4]:
        st.write(f"- {row}")


def _wave_path_callout(height_m: float) -> str:
    if height_m >= 3.0:
        return "High-impact surf path; prioritize protective planning."
    if height_m >= 2.0:
        return "Elevated wave path; monitor shoreline operations closely."
    if height_m >= 1.2:
        return "Moderate regime; conditions can pivot with wind/tide shifts."
    return "Low regime path; near-term stress likely limited."


def render_wave_monte_carlo_panel(report: WaveMonteCarloReport) -> None:
    st.subheader("Wave Monte Carlo Forecast Agent")
    if report.mc_hypothesis_ready:
        st.success("Monte Carlo ADK Interpretation: Ready")
    else:
        st.warning("Monte Carlo Hypothesis blocked: ADK interpretation required")
        if report.mc_adk_attempts:
            st.caption("ADK attempts: " + "; ".join(report.mc_adk_attempts))

    st.markdown(f"**Simulation Headline:** {report.thesis}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final P50 (m)", f"{report.final_p50_m:.2f}")
    m2.metric("Final P90 (m)", f"{report.final_p90_m:.2f}")
    m3.metric("P(end > 2m)", f"{report.probability_exceed_2m * 100:.1f}%")
    m4.metric("P(any > 3m)", f"{report.probability_reach_3m_anytime * 100:.1f}%")

    st.caption(
        f"Model: {report.model_name} | drift={report.drift_per_hour:+.4f}/h | "
        f"vol={report.volatility_per_sqrt_hour:.4f} | kappa={report.mean_reversion_kappa:.4f} | "
        f"jump_intensity={report.jump_intensity_per_hour:.4f}/h | t_df={report.student_t_df:.2f}"
    )
    adv1, adv2, adv3, adv4 = st.columns(4)
    adv1.metric("Expected Peak (m)", f"{report.expected_peak_m:.2f}")
    adv2.metric("Tail Risk CVaR95", f"{report.cvar95_final_m:.2f}")
    adv3.metric("Storm Persistence", f"{report.storm_regime_persistence:.2f}")
    adv4.metric("Storm Vol Mult", f"{report.storm_vol_multiplier:.2f}")

    st.markdown("**Monte Carlo Interpretation Status**")
    status_cols = st.columns(3)
    with status_cols[0]:
        st.metric("MC ADK Gate", report.mc_adk_status.replace("_", " ").title())
    with status_cols[1]:
        st.metric("Narrative Ready", "Yes" if report.mc_hypothesis_ready else "No")
    with status_cols[2]:
        st.metric("ADK Attempts", str(len(report.mc_adk_attempts)))

    if report.mc_adk_attempts:
        st.caption("Attempts: " + "; ".join(report.mc_adk_attempts))

    if report.mc_hypothesis_ready and report.mc_adk_hypothesis:
        st.markdown(f"**Monte Carlo Hypothesis:** {report.mc_adk_hypothesis}")
        if report.plain_english_summary.strip():
            st.info(report.plain_english_summary.strip())
        if report.post_mc_hypothesis:
            st.markdown("**After-Simulation Hypothesis Update**")
            st.success(report.post_mc_hypothesis)
    else:
        st.warning("Monte Carlo narrative is still pending ADK interpretation. Simulation stats below are valid.")

    quantiles = pd.DataFrame(report.trajectory_quantiles)
    fig = go.Figure()
    if not quantiles.empty:
        fig.add_trace(
            go.Scatter(
                x=quantiles["hour"],
                y=quantiles["p90_m"],
                mode="lines",
                line={"color": "rgba(255, 159, 28, 0.0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=quantiles["hour"],
                y=quantiles["p10_m"],
                mode="lines",
                fill="tonexty",
                name="P10-P90 Envelope",
                line={"color": "rgba(46, 196, 182, 0.0)"},
                fillcolor="rgba(46, 196, 182, 0.25)",
                hovertemplate="Hour %{x}<br>P10: %{y:.2f} m<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=quantiles["hour"],
                y=quantiles["p50_m"],
                mode="lines+markers",
                name="Median Path",
                line={"color": "#6ad4ff", "width": 2.5},
                marker={"size": 5},
                text=[_wave_path_callout(float(v)) for v in quantiles["p50_m"]],
                hovertemplate="Hour %{x}<br>Median: %{y:.2f} m<br>%{text}<extra></extra>",
            )
        )

    for idx, path in enumerate(report.sample_paths[:6]):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(path))),
                y=path,
                mode="lines",
                name=f"Sample Path {idx+1}",
                line={"color": "rgba(255,255,255,0.20)", "width": 1},
                hovertemplate="Hour %{x}<br>Path height: %{y:.2f} m<br>%{text}<extra></extra>",
                text=[_wave_path_callout(float(v)) for v in path],
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Wave Height Monte Carlo Paths",
        xaxis_title="Forecast Hour",
        yaxis_title="Wave Height (m)",
        template="plotly_dark",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified",
        paper_bgcolor="#040710",
        plot_bgcolor="#040710",
        font={"color": "#eaf6ff"},
    )
    st.plotly_chart(fig, use_container_width=True)

    hist_col, curve_col = st.columns(2)
    final_sample = pd.Series(report.final_state_sample, dtype=float) if report.final_state_sample else pd.Series(dtype=float)

    with hist_col:
        hist_fig = go.Figure()
        if not final_sample.empty:
            hist_fig.add_trace(
                go.Histogram(
                    x=final_sample,
                    nbinsx=36,
                    name="Final Wave Height Distribution",
                    marker={"color": "rgba(106, 212, 255, 0.75)"},
                    hovertemplate="Final wave: %{x:.2f} m<br>Count: %{y}<extra></extra>",
                )
            )
            hist_fig.add_vline(x=2.0, line_dash="dash", line_color="#ffd166")
            hist_fig.add_vline(x=3.0, line_dash="dash", line_color="#ff6b6b")
        hist_fig.update_layout(
            title="Final-State Distribution",
            xaxis_title="Final Wave Height (m)",
            yaxis_title="Path Count",
            template="plotly_dark",
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            paper_bgcolor="#040710",
            plot_bgcolor="#040710",
            font={"color": "#eaf6ff"},
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with curve_col:
        curve_fig = go.Figure()
        if not final_sample.empty:
            thresholds = np.arange(0.5, 4.6, 0.1)
            exceed = [float((final_sample >= th).mean()) * 100 for th in thresholds]
            curve_fig.add_trace(
                go.Scatter(
                    x=thresholds,
                    y=exceed,
                    mode="lines+markers",
                    line={"color": "#2ec4b6", "width": 2.5},
                    marker={"size": 4},
                    hovertemplate="Threshold: %{x:.1f} m<br>P(final >= threshold): %{y:.1f}%<extra></extra>",
                    name="Exceedance Curve",
                )
            )
        curve_fig.update_layout(
            title="Threshold Exceedance Curve",
            xaxis_title="Wave Threshold (m)",
            yaxis_title="Probability (%)",
            template="plotly_dark",
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            paper_bgcolor="#040710",
            plot_bgcolor="#040710",
            font={"color": "#eaf6ff"},
        )
        st.plotly_chart(curve_fig, use_container_width=True)

    st.markdown("**Interpretation**")
    for row in report.interpretation_points:
        st.write(f"- {row}")
    if report.limitations:
        st.markdown("**Simulation caveats**")
        for row in report.limitations[:4]:
            st.write(f"- {row}")


def thesis_source_label(result: RunResult) -> str:
    if result.adk_status in {"success", "recovery_success"}:
        return "ADK Multi-Agent"
    if result.adk_status == "failed_transient_fallback":
        return "Deterministic Fallback (ADK failed transiently)"
    return "Deterministic Fallback"


def dataframe_download_button(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    st.download_button(
        label=f"Download {name} CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{name}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="OceanWatch",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_ocean_theme()
    service = get_service()

    st.markdown(
        """
        <div class="ocean-hero">
          <h2 style="margin:0;">OceanWatch: Multi-Agent Coastal Analyst</h2>
          <p style="margin:0.35rem 0 0 0;">
            NOAA runtime analytics + ADK orchestration for U.S. coastal intelligence.
            Includes severity scoring, regressions, likelihoods, and simulation outcomes.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis Controls")
        labels = [f"{entry.display_name} ({entry.coast})" for entry in STATION_CATALOG]
        selected_label = st.selectbox("Coastal Station", labels, index=0)
        station = get_station_by_label(selected_label)

        hours_back = st.slider(
            "Analysis Horizon (hours)",
            min_value=24,
            max_value=336,
            value=settings.default_hours_back,
            step=24,
        )
        units = st.selectbox("Units", ["metric", "english"], index=0)
        analysis_focus = st.selectbox(
            "Simulation Focus",
            ["balanced", "storm_readiness", "calm_operations"],
            index=0,
        )
        question = st.text_area(
            "Question for OceanWatch",
            value=(
                "What is happening at this station right now, what irregular patterns are present, "
                "and what do likely next-tide scenarios imply?"
            ),
            height=120,
        )
        run_clicked = st.button("Run Multi-Agent Analysis", type="primary")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if run_clicked:
        with st.spinner("Collecting NOAA data, running multi-agent ADK pipeline, generating hypothesis..."):
            request = AnalysisRequest(
                user_question=question,
                station_key=station.key,
                hours_back=hours_back,
                units=units,
                analysis_focus=analysis_focus,
            )
            result = service.run_analysis(request)
            st.session_state.last_result = result
            st.session_state.last_wave_mc = None
            st.session_state.chat_history.append((question, result.insight.thesis))

    result = st.session_state.get("last_result")
    mc_report = st.session_state.get("last_wave_mc")
    st.subheader("OceanWatch Chat")
    for user_msg, assistant_msg in st.session_state.chat_history[-3:]:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(assistant_msg)

    if result is None:
        st.info("Run an analysis from the sidebar to unlock executive dashboards, quant visuals, and Monte Carlo handoff.")
        st.subheader("Station Map")
        render_station_map(station.key, result=None)
        return

    st.markdown(
        (
            "<div class='kpi-ribbon'><b>Executive Snapshot</b>: "
            f"{result.metrics.advanced_analytics.get('risk_headline', 'Awaiting analysis headline.')}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    render_source_health(result.health)
    render_metric_cards(result, mc_report=mc_report)
    st.info(result.metrics.confidence_note)
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    tabs = st.tabs(
        [
            "Executive Brief",
            "Geo Lens",
            "Quant Lab",
            "Monte Carlo Lab",
            "Data Room",
        ]
    )
    tab_synthesis = build_tab_hypothesis_synthesis(result, mc_report=mc_report)

    with tabs[0]:
        render_tab_hypothesis_synthesis("Executive Brief", tab_synthesis["Executive Brief"], result.insight.thesis)
        col_left, col_right = st.columns([1.55, 1], gap="large")
        with col_left:
            render_agent_workflow(result, mc_report=mc_report)
            st.subheader("Final Hypothesis")
            source_label = thesis_source_label(result)
            if source_label == "ADK Multi-Agent":
                st.success(f"Hypothesis Source: {source_label}")
            else:
                st.warning(f"Hypothesis Source: {source_label}")
            st.markdown(f"**Hypothesis:** {result.insight.thesis}")
            if result.insight.narrative_paragraphs:
                for paragraph in result.insight.narrative_paragraphs[:3]:
                    st.write(paragraph)
            st.markdown("**Evidence bullets**")
            for bullet in result.insight.evidence_bullets:
                st.write(f"- {bullet}")
            if result.insight.notable_anomalies:
                st.markdown("**Notable anomalies**")
                for item in result.insight.notable_anomalies:
                    st.write(f"- {item}")
            st.markdown("**Limitations**")
            for item in result.insight.limitations:
                st.write(f"- {item}")
            st.markdown("**Recommended follow-ups**")
            for item in result.insight.recommended_followups:
                st.write(f"- {item}")
        with col_right:
            st.subheader("Station Map")
            render_station_map(station.key, result=result)
            st.subheader("Station Metadata")
            st.json(result.station)
            st.markdown(
                f"**ADK status:** `{result.adk_status}`"
                + f" | `attempted: {result.adk_attempted}`"
                + (f" | `model: {result.adk_model_used}`" if result.adk_model_used else "")
                + (f" | `{result.adk_error_summary}`" if result.adk_error_summary else "")
            )

    with tabs[1]:
        render_tab_hypothesis_synthesis("Geo Lens", tab_synthesis["Geo Lens"], result.insight.thesis)
        render_ocean_lens(result.station, result)

    with tabs[2]:
        render_tab_hypothesis_synthesis("Quant Lab", tab_synthesis["Quant Lab"], result.insight.thesis)
        render_advanced_panels(result)
        st.subheader("Visual Analytics")
        for name, figure in result.figures.items():
            st.plotly_chart(figure, use_container_width=True)

    with tabs[3]:
        render_tab_hypothesis_synthesis("Monte Carlo Lab", tab_synthesis["Monte Carlo Lab"], result.insight.thesis)
        st.subheader("Monte Carlo Follow-up (Agent Handoff)")
        st.caption(
            "After preliminary analysis, choose horizon and path count, then hand off to the Monte Carlo specialist agent."
        )
        mc_days = st.slider(
            "Monte Carlo Horizon (days)",
            min_value=1,
            max_value=7,
            value=3,
            step=1,
            key="mc_days_after_analysis",
        )
        mc_paths = st.slider(
            "Monte Carlo Paths",
            min_value=300,
            max_value=3000,
            value=900,
            step=100,
            key="mc_paths_after_analysis",
        )
        st.info(
            "Agent handoff: CoordinatorAgent -> WaveMonteCarloCoordinator "
            "using runtime NOAA context + advanced stochastic simulation."
        )
        run_mc_clicked = st.button("Run Monte Carlo Handoff", key="run_wave_mc")
        if run_mc_clicked:
            with st.spinner("Running wave-path Monte Carlo specialist agent..."):
                st.session_state.last_wave_mc = service.run_wave_monte_carlo_agent(
                    result,
                    days_ahead=mc_days,
                    path_count=mc_paths,
                )
                mc_report = st.session_state.last_wave_mc
        if mc_report is not None:
            render_wave_monte_carlo_panel(mc_report)

    with tabs[4]:
        render_tab_hypothesis_synthesis("Data Room", tab_synthesis["Data Room"], result.insight.thesis)
        st.subheader("Data Exports")
        data_cols = st.columns(3)
        with data_cols[0]:
            dataframe_download_button("buoy_observations", result.tables["buoy"])
        with data_cols[1]:
            dataframe_download_button("water_levels", result.tables["water_level"])
        with data_cols[2]:
            dataframe_download_button("tide_predictions", result.tables["tide_predictions"])

        st.markdown("**Dataset previews**")
        preview_tabs = st.tabs(["Buoy", "Water Level", "Tide Predictions"])
        with preview_tabs[0]:
            st.dataframe(result.tables["buoy"].tail(100), use_container_width=True)
        with preview_tabs[1]:
            st.dataframe(result.tables["water_level"].tail(100), use_container_width=True)
        with preview_tabs[2]:
            st.dataframe(result.tables["tide_predictions"].tail(100), use_container_width=True)


if __name__ == "__main__":
    main()

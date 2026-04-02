from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oceanwatch.config import load_settings
from oceanwatch.schemas import AnalysisRequest, RunResult, WaveMonteCarloReport
from oceanwatch.service import OceanWatchService
from oceanwatch.stations import STATION_CATALOG, get_station_by_label


settings = load_settings()


@st.cache_resource
def get_service() -> OceanWatchService:
    return OceanWatchService(settings=settings)


def apply_ocean_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 12%, rgba(70, 160, 228, 0.30), transparent 32%),
                radial-gradient(circle at 92% 14%, rgba(30, 220, 202, 0.22), transparent 36%),
                linear-gradient(180deg, #021427 0%, #052a45 52%, #0a3b5e 100%);
            color: #ecf7ff;
        }
        [data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #032339 0%, #0d466c 100%);
            border-right: 1px solid rgba(120, 204, 255, 0.20);
        }
        h1, h2, h3, h4, h5, h6, p, li, label, div, span { color: #ecf7ff; }
        [data-testid="stAlertContainer"] > div { border-radius: 12px; }
        [data-testid="stChatMessage"] {
            background: rgba(7, 54, 84, 0.45);
            border: 1px solid rgba(120, 204, 255, 0.22);
            border-radius: 12px;
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
        height=340,
        margin={"l": 0, "r": 0, "t": 38, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.06, "x": 0},
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


def _ocean_thumbnail_catalog(station: dict[str, object]) -> list[dict[str, str]]:
    coast = str(station.get("coast", "West Coast"))
    by_coast: dict[str, list[dict[str, str]]] = {
        "West Coast": [
            {
                "title": "California Kelp Coast",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Big_Sur_Coastline.jpg/640px-Big_Sur_Coastline.jpg",
                "source": "Wikimedia Commons",
            },
            {
                "title": "Pacific Swell",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Waves_in_the_Pacific_Ocean.jpg/640px-Waves_in_the_Pacific_Ocean.jpg",
                "source": "Wikimedia Commons",
            },
            {
                "title": "NOAA Ocean Explorer",
                "url": "https://oceanexplorer.noaa.gov/oceanexplorer-edu/images/ocean_image_gallery/cover.jpg",
                "source": "NOAA Ocean Explorer",
            },
        ],
        "East Coast": [
            {
                "title": "Atlantic Coastline",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Atlantic_Ocean_coastline.jpg/640px-Atlantic_Ocean_coastline.jpg",
                "source": "Wikimedia Commons",
            },
            {
                "title": "NOAA Coast Survey",
                "url": "https://nauticalcharts.noaa.gov/images/hero/charts-hero.jpg",
                "source": "NOAA Coast Survey",
            },
            {
                "title": "Coastal Storm Seas",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Atlantic_waves.jpg/640px-Atlantic_waves.jpg",
                "source": "Wikimedia Commons",
            },
        ],
        "Atlantic": [
            {
                "title": "Atlantic Open Water",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Atlantic_Ocean.jpg/640px-Atlantic_Ocean.jpg",
                "source": "Wikimedia Commons",
            },
            {
                "title": "NOAA Marine Ops",
                "url": "https://www.noaa.gov/sites/default/files/styles/card_medium/public/2023-08/NOAA-ship-ocean.jpg",
                "source": "NOAA",
            },
            {
                "title": "Shoreline Dynamics",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Ocean_waves.jpg/640px-Ocean_waves.jpg",
                "source": "Wikimedia Commons",
            },
        ],
        "Gulf": [
            {
                "title": "Gulf Coast Waters",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Gulf_of_Mexico_satellite.jpg/640px-Gulf_of_Mexico_satellite.jpg",
                "source": "Wikimedia Commons",
            },
            {
                "title": "NOAA Coastal Resilience",
                "url": "https://oceanservice.noaa.gov/facts/coastal/images/coastal-main.jpg",
                "source": "NOAA Ocean Service",
            },
            {
                "title": "Nearshore Surf",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Ocean_surf.jpg/640px-Ocean_surf.jpg",
                "source": "Wikimedia Commons",
            },
        ],
        "Pacific": [
            {
                "title": "Pacific Basin",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Pacific_Ocean_-_en.png/640px-Pacific_Ocean_-_en.png",
                "source": "Wikimedia Commons",
            },
            {
                "title": "Island Swell",
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Pacific_ocean_waves.jpg/640px-Pacific_ocean_waves.jpg",
                "source": "Wikimedia Commons",
            },
            {
                "title": "NOAA Ocean Service",
                "url": "https://oceanservice.noaa.gov/news/images/ocean-facts.jpg",
                "source": "NOAA Ocean Service",
            },
        ],
    }
    return by_coast.get(coast, by_coast["West Coast"])


def render_ocean_thumbnails(station: dict[str, object]) -> None:
    st.subheader("Ocean Lens")
    st.caption("NOAA/public ocean thumbnails mapped to the selected coast.")
    thumbs = _ocean_thumbnail_catalog(station)
    cols = st.columns(3)
    for col, thumb in zip(cols, thumbs):
        with col:
            st.markdown(
                f"[![{thumb['title']}]({thumb['url']})]({thumb['url']})",
                unsafe_allow_html=False,
            )
            st.caption(f"{thumb['title']} · {thumb['source']}")


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
        show_cols = [
            "label",
            "slope_per_hour",
            "r_squared",
            "p_value",
            "projected_6h",
            "threshold",
            "threshold_units",
            "exceedance_probability",
            "risk_band",
        ]
        existing = [col for col in show_cols if col in reg_df.columns]
        st.dataframe(reg_df[existing], use_container_width=True)

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
        st.dataframe(sim_df, use_container_width=True)


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
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified",
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
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
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
            template="plotly_white",
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
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
    st.set_page_config(page_title="OceanWatch", page_icon="🌊", layout="wide")
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

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.subheader("OceanWatch Chat")
        for user_msg, assistant_msg in st.session_state.chat_history[-5:]:
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(assistant_msg)

        if result is not None:
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

            render_advanced_panels(result)

            st.markdown("---")
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

    with right:
        st.subheader("Station Map")
        render_station_map(station.key, result=result)

        if result is not None:
            st.subheader("Station Metadata")
            st.json(result.station)
            st.markdown(
                f"**ADK status:** `{result.adk_status}`"
                + f" | `attempted: {result.adk_attempted}`"
                + (f" | `model: {result.adk_model_used}`" if result.adk_model_used else "")
                + (f" | `{result.adk_error_summary}`" if result.adk_error_summary else "")
            )
            render_source_health(result.health)

            if result.warnings:
                st.subheader("Warnings")
                for warning in result.warnings:
                    st.warning(warning)

            render_metric_cards(result, mc_report=mc_report)
            st.info(result.metrics.confidence_note)
            render_ocean_thumbnails(result.station)

    if result is not None:
        st.subheader("Visual Analytics")
        for name, figure in result.figures.items():
            st.plotly_chart(figure, use_container_width=True)

        st.subheader("Data Exports")
        data_cols = st.columns(3)
        with data_cols[0]:
            dataframe_download_button("buoy_observations", result.tables["buoy"])
        with data_cols[1]:
            dataframe_download_button("water_levels", result.tables["water_level"])
        with data_cols[2]:
            dataframe_download_button("tide_predictions", result.tables["tide_predictions"])


if __name__ == "__main__":
    main()

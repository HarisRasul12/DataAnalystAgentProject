from __future__ import annotations

from typing import Any

from oceanwatch.config import Settings
from oceanwatch.schemas import (
    BuoyAgentReport,
    InsightReport,
    PostMonteCarloHypothesis,
    RiskAgentReport,
    TideAgentReport,
    WaveMonteCarloReport,
    WaveHypothesisInterpretation,
    WaveSimulationSnapshot,
)


def adk_is_available() -> bool:
    try:
        import google.adk.agents  # noqa: F401
        import google.adk.runners  # noqa: F401
        import google.adk.sessions  # noqa: F401
        import google.genai.types  # noqa: F401
    except Exception:
        return False
    return True


def create_oceanwatch_root_agent(settings: Settings, tools: list[Any], model_name: str | None = None) -> Any:
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

    tool_lookup = {tool.__name__: tool for tool in tools}
    model = model_name or settings.vertex_model

    buoy_agent = LlmAgent(
        name="BuoyAgent",
        model=model,
        description="Fetches and analyzes NOAA NDBC buoy conditions.",
        instruction=(
            "Role: senior physical-ocean diagnostics analyst focused on buoy dynamics.\n"
            "Protocol:\n"
            "1) Call get_ndbc_observations first.\n"
            "2) Call compute_ocean_metrics with source='buoy'.\n"
            "3) Extract statistically meaningful signals only: central tendency, trend slope, anomaly density.\n"
            "4) Separate stable regime vs regime-shift evidence using concrete numbers.\n"
            "Output requirements:\n"
            "- status='ok' when analysis is complete.\n"
            "- highlights: 4-7 bullets with explicit metric values and units.\n"
            "- caveats: data-quality caveats, sample-size limits, and uncertainty notes.\n"
            "Constraints:\n"
            "- Use runtime tools only, no fabricated observations.\n"
            "- Every key claim must include at least one numeric anchor."
        ),
        tools=[tool_lookup["get_ndbc_observations"], tool_lookup["compute_ocean_metrics"]],
        output_schema=BuoyAgentReport,
        output_key="buoy_report",
    )

    tide_agent = LlmAgent(
        name="TideAgent",
        model=model,
        description="Fetches and analyzes NOAA CO-OPS water levels and tide predictions.",
        instruction=(
            "Role: coastal hydrodynamics analyst for observed levels and tide-cycle structure.\n"
            "Protocol:\n"
            "1) Call get_coops_water_level and get_coops_tide_predictions.\n"
            "2) Call compute_ocean_metrics with source='coops'.\n"
            "3) Quantify water-level trajectory and tide context (range, high/low cadence).\n"
            "4) Flag whether tidal phase likely amplifies or dampens risk right now.\n"
            "Output requirements:\n"
            "- highlights must include observed water-level numbers, slope direction, and tide context.\n"
            "- caveats must identify missing feed risk or temporal coverage limits.\n"
            "Constraints:\n"
            "- No speculative coastal flooding claims without explicit data support."
        ),
        tools=[
            tool_lookup["get_coops_water_level"],
            tool_lookup["get_coops_tide_predictions"],
            tool_lookup["compute_ocean_metrics"],
        ],
        output_schema=TideAgentReport,
        output_key="tide_report",
    )

    risk_agent = LlmAgent(
        name="RiskAgent",
        model=model,
        description="Runs severity scoring, regressions, likelihoods, and scenario simulations.",
        instruction=(
            "Role: quantitative coastal-risk modeler.\n"
            "Protocol:\n"
            "1) Call compute_risk_analytics.\n"
            "2) Call simulate_coastal_scenarios.\n"
            "3) Convert model outputs into interpretable risk framing.\n"
            "Reasoning requirements:\n"
            "- Explicitly connect regression slope direction, exceedance probabilities, and irregular patterns.\n"
            "- Distinguish baseline risk vs tail risk.\n"
            "- Prioritize interpretable decision language while preserving statistical fidelity.\n"
            "Output requirements:\n"
            "- severity_level should map to quantitative outputs.\n"
            "- likelihood_summary should include at least one probability.\n"
            "- highlights should include 4-7 evidence bullets grounded in computed analytics.\n"
            "- caveats should include model assumptions and uncertainty boundaries."
        ),
        tools=[tool_lookup["compute_risk_analytics"], tool_lookup["simulate_coastal_scenarios"]],
        output_schema=RiskAgentReport,
        output_key="risk_report",
    )

    pattern_agent = LlmAgent(
        name="PatternHypothesisAgent",
        model=model,
        description="Builds cross-source pattern interpretation before final synthesis.",
        instruction=(
            "Role: cross-signal pattern analyst.\n"
            "Protocol:\n"
            "1) Call compute_ocean_metrics with source='all'.\n"
            "2) Call compute_risk_analytics.\n"
            "3) Identify the strongest cross-source pattern linking buoy, water level, and risk indicators.\n"
            "Output requirements:\n"
            "- status='ok'.\n"
            "- highlights: include at least one 'because' chain linking multiple metrics.\n"
            "- caveats: include at least one alternate explanation and one uncertainty statement.\n"
            "Constraints:\n"
            "- No generic summary; surface a specific, testable pattern."
        ),
        tools=[tool_lookup["compute_ocean_metrics"], tool_lookup["compute_risk_analytics"]],
        output_schema=BuoyAgentReport,
        output_key="pattern_report",
    )

    parallel_data = ParallelAgent(
        name="ParallelOceanCollection",
        sub_agents=[buoy_agent, tide_agent, risk_agent, pattern_agent],
    )

    hypothesis_builder_agent = LlmAgent(
        name="HypothesisBuilderAgent",
        model=model,
        description="Builds a rigorous data-grounded hypothesis from specialist reports.",
        instruction=(
            "Role: lead coastal intelligence analyst creating a hypothesis, not a generic summary.\n"
            "Inputs: {buoy_report}, {tide_report}, {risk_report}, {pattern_report}.\n"
            "Method:\n"
            "1) Propose one primary hypothesis explaining current coastal state.\n"
            "2) Write 2-3 narrative_paragraphs that connect metrics to mechanism in plain but technical language.\n"
            "3) Support it with multi-source evidence bullets containing concrete numbers.\n"
            "4) Explicitly state at least one alternative hypothesis that was considered but weaker.\n"
            "5) Include anomalies that materially affect confidence.\n"
            "Output schema: InsightReport.\n"
            "Style constraints:\n"
            "- Write in clear but technically strong language.\n"
            "- Make causal links explicit ('X suggests Y because Z').\n"
            "- No unsupported claims."
        ),
        tools=[],
        output_schema=InsightReport,
        output_key="hypothesis_draft",
    )

    synthesis_agent = LlmAgent(
        name="HypothesisCriticRefinerAgent",
        model=model,
        description="Critiques and refines the hypothesis for evidentiary rigor and clarity.",
        instruction=(
            "Role: skeptical reviewer and scientific editor.\n"
            "Inputs: {hypothesis_draft}, {buoy_report}, {tide_report}, {risk_report}, {pattern_report}.\n"
            "Tasks:\n"
            "1) Stress-test the draft for weak evidence links and overclaiming.\n"
            "2) Strengthen the final hypothesis language so it is precise, testable, and data-grounded.\n"
            "3) Ensure narrative_paragraphs are cohesive, analytical, and cite quantitative evidence.\n"
            "4) Ensure evidence bullets are specific, numeric, and understandable for non-experts.\n"
            "5) Ensure limitations and follow-ups are actionable.\n"
            "Return only InsightReport in the final output_key insight_report."
        ),
        tools=[],
        output_schema=InsightReport,
        output_key="insight_report",
    )

    return SequentialAgent(
        name="CoordinatorAgent",
        sub_agents=[parallel_data, hypothesis_builder_agent, synthesis_agent],
    )


def create_wave_monte_carlo_agent(settings: Settings, tools: list[Any], model_name: str | None = None) -> Any:
    from google.adk.agents import LlmAgent, SequentialAgent

    tool_lookup = {tool.__name__: tool for tool in tools}
    model = model_name or settings.vertex_model

    simulation_agent = LlmAgent(
        name="WaveSimulationAgent",
        model=model,
        description="Loads deterministic Monte Carlo simulation stats via tool call.",
        instruction=(
            "Role: deterministic simulation loader.\n"
            "Protocol:\n"
            "1) Call load_simulation_snapshot exactly once.\n"
            "2) Return the snapshot unchanged.\n"
            "Constraints:\n"
            "- Do not invent values.\n"
            "- Output must match WaveSimulationSnapshot."
        ),
        tools=[tool_lookup["load_simulation_snapshot"]],
        output_schema=WaveSimulationSnapshot,
        output_key="simulation_snapshot",
    )

    hypothesis_agent = LlmAgent(
        name="WaveHypothesisAgent",
        model=model,
        description="Creates ADK-only interpretation hypothesis from simulation snapshot and runtime context.",
        instruction=(
            "Role: Monte Carlo interpretation analyst.\n"
            "Inputs: {simulation_snapshot}, {wave_context}.\n"
            "Tasks:\n"
            "1) Produce mc_adk_hypothesis grounded in percentiles/probabilities and tail-risk metrics.\n"
            "2) Produce plain_english_summary in concise operational language.\n"
            "3) Include interpretation_points and limitations with uncertainty notes.\n"
            "Constraints:\n"
            "- Must cite simulation numbers explicitly.\n"
            "- No generic language.\n"
            "- Do not call external tools.\n"
            "- Output must match WaveHypothesisInterpretation."
        ),
        tools=[],
        output_schema=WaveHypothesisInterpretation,
        output_key="wave_hypothesis_interpretation",
    )

    post_hypothesis_agent = LlmAgent(
        name="PostMonteCarloHypothesisAgent",
        model=model,
        description="Compares preliminary hypothesis with Monte Carlo interpretation and produces after-simulation hypothesis.",
        instruction=(
            "Role: hypothesis reconciler.\n"
            "Inputs: {preliminary_hypothesis}, {wave_hypothesis_interpretation}, {simulation_snapshot}.\n"
            "Tasks:\n"
            "1) Write post_mc_hypothesis explaining what changed after simulation evidence.\n"
            "2) Add 2-4 what_changed bullets that contrast pre-vs-post understanding.\n"
            "3) Keep language clear and evidence-linked.\n"
            "Output must match PostMonteCarloHypothesis."
        ),
        tools=[],
        output_schema=PostMonteCarloHypothesis,
        output_key="post_mc_hypothesis_report",
    )

    return SequentialAgent(
        name="WaveMonteCarloCoordinator",
        sub_agents=[simulation_agent, hypothesis_agent, post_hypothesis_agent],
    )

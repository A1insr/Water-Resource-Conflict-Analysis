"""
Pipeline for Chapter 6 (Game analysis):
1) Build criterion scores for all 18 scenarios using criteria-functions module.
2) Compute AHP weights (kept exactly from the original pipeline).
3) Aggregate scenario utilities per player.
4) Build and display payoff tables:
   - Two 3×3 tables (Isfahan × Government) conditioned on Yazd strategy (S_Y / A_Y).

This file is intended to *replace* the manual scenario-by-scenario scoring in the old pipeline.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
import os

import pandas as pd

from ahp_utils import ahp_weights, compute_utility, normalize_to_minus5_plus5
import payoff_table as pt

# IMPORTANT: import the exact criteria-functions file you are using in your thesis
import criteria_functions_UPDATED_optionA as cf


# -------------------------------------------------------------------
# SCENARIOS (3 × 3 × 2 = 18)
# Naming: I (Isfahan), G (Government), Y (Yazd)
# -------------------------------------------------------------------
SCENARIOS = [
    "S_I_C_G_S_Y", "S_I_C_G_A_Y",
    "S_I_Q_G_S_Y", "S_I_Q_G_A_Y",
    "S_I_R_G_S_Y", "S_I_R_G_A_Y",

    "M_I_C_G_S_Y", "M_I_C_G_A_Y",
    "M_I_Q_G_S_Y", "M_I_Q_G_A_Y",
    "M_I_R_G_S_Y", "M_I_R_G_A_Y",

    "D_I_C_G_S_Y", "D_I_C_G_A_Y",
    "D_I_Q_G_S_Y", "D_I_Q_G_A_Y",
    "D_I_R_G_S_Y", "D_I_R_G_A_Y"
]


# -------------------------------------------------------------------
# CRITERIA LISTS (must match names used in AHP and thesis)
# -------------------------------------------------------------------
criteria_I = [
    "agricultural_income",
    "social_satisfaction",
    "environment"
]

criteria_Y = [
    "urban_industrial_water_security",
    "conflict_escalation_cost",
    "justice_feeling"
]

criteria_G = [
    "budget_cost",        # higher score = lower cost / better for gov
    "economic_growth",
    "public_satisfaction",
    "security_risk"       # higher score = safer / lower risk
]


# -------------------------------------------------------------------
# AHP PAIRWISE MATRICES (EXPERT-BASED) — KEPT EXACTLY AS ORIGINAL
# -------------------------------------------------------------------
pairwise_I = [
    [1,   2,   3],
    [1/2, 1,   2],
    [1/3, 1/2, 1]
]

# NOTE: now interpreted as:
# 1) urban_industrial_water_security
# 2) conflict_escalation_cost
# 3) justice_feeling
pairwise_Y = [
    [1,   3,   3],
    [1/3, 1,   1/2],
    [1/3, 2, 1]
]

pairwise_G = [
    [1,   4,   3,   1/3],
    [1/4, 1,   1/2, 1/6],
    [1/3, 2,   1,   1/5],
    [3,   6,   5,   1]
]


# -------------------------------------------------------------------
# Data classes for passing inputs cleanly
# -------------------------------------------------------------------
HydroState = Literal["dry", "normal", "wet"]


@dataclass
class ScoreBuildersConfig:
    """
    Configuration to build all criterion score dictionaries.

    You can run fully with defaults, BUT:
    - agricultural_income (Isfahan) requires agri_params.
      Either provide agri_params directly, OR provide spi_df + area_df to calibrate.
    """
    hydro_state: HydroState = "normal"

    # ---- Isfahan agri data/calibration ----
    spi_df: Optional[pd.DataFrame] = None
    area_df: Optional[pd.DataFrame] = None
    agri_params: Optional[cf.AgriIsfahanParams] = None

    # ---- Optional excel inputs for government criteria (will fall back to defaults if None) ----
    budget_cost_excel: Optional[str] = None
    economic_growth_excel: Optional[str] = None
    public_satisfaction_excel: Optional[str] = None
    security_risk_excel: Optional[str] = None


def _ensure_agri_params(cfg: ScoreBuildersConfig) -> cf.AgriIsfahanParams:
    if cfg.agri_params is not None:
        return cfg.agri_params

    if cfg.spi_df is None or cfg.area_df is None:
        raise ValueError(
            "To build agricultural_income scores you must provide either:\n"
            "  - cfg.agri_params, OR\n"
            "  - cfg.spi_df and cfg.area_df to calibrate_agri_isfahan()."
        )

    return cf.calibrate_agri_isfahan(cfg.spi_df, cfg.area_df)


def build_all_scores(
    cfg: ScoreBuildersConfig
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]]
]:
    """
    Build the complete score dictionaries (scenario -> {criterion: score}) for:
      - Isfahan
      - Yazd
      - Government

    Returns
    -------
    (scores_I, scores_Y, scores_G)
    where each is: dict[scenario] -> dict[criterion] -> score in [0,100]
    """
    # -------------------- Isfahan --------------------
    agri_params = _ensure_agri_params(cfg)

    agri_scores = cf.build_agri_scores_isfahan_for_scenarios(
        agri_params, hydro_state=cfg.hydro_state, include_yazd_dimension=True
    )

    # ---- Social satisfaction (Isfahan) with your trade-off injection ----
    social_params = cf.default_social_isfahan_params()

    social_params.contentment_effect["S_I"] = 0
    social_params.contentment_effect["M_I"] = 2
    social_params.contentment_effect["D_I"] = 6
    social_params.contentment_effect["A_Y"] = -2
    social_params.contentment_effect["Q_G"] = -7
    social_params.contentment_effect["C_G"] = 4
    social_params.contentment_effect["R_G"] = 6

    social_params.justice_effect["C_G"] = 2
    social_params.justice_effect["R_G"] = 4

    social_params.crisis_effect["M_I"] = -12
    social_params.crisis_effect["D_I"] = -7
    social_params.crisis_effect["A_Y"] = -3
    social_params.crisis_effect["Q_G"] = -10
    social_params.crisis_effect["C_G"] = 5
    social_params.crisis_effect["R_G"] = 8

    social_scores = cf.build_social_scores_isfahan_for_scenarios(social_params)
    env_scores = cf.build_env_scores_isfahan_for_scenarios(cf.default_env_isfahan_params())

    scores_I: Dict[str, Dict[str, float]] = {}
    for s in SCENARIOS:
        scores_I[s] = {
            "agricultural_income": float(agri_scores[s]),
            "social_satisfaction": float(social_scores[s]),
            "environment": float(env_scores[s]),
        }

    # -------------------- Yazd (Option A) --------------------
    ui_params = cf.default_urban_industrial_water_security_yazd_params()
    ui_scores = cf.build_urban_industrial_water_security_scores_yazd_for_scenarios(
        ui_params, hydro_state=cfg.hydro_state
    )

    # Keep industry params for Government economic growth calculation (E2) — not used as Yazd payoff criterion
    ind_params = cf.default_industry_yazd_params()

    cost_params = cf.default_conflict_escalation_cost_yazd_params()
    cost_scores = cf.build_conflict_escalation_cost_scores_yazd_for_scenarios(cost_params)

    jus_scores = cf.build_perceived_justice_scores_yazd_for_scenarios(cf.default_justice_yazd_params())

    scores_Y: Dict[str, Dict[str, float]] = {}
    for s in SCENARIOS:
        scores_Y[s] = {
            "urban_industrial_water_security": float(ui_scores[s]),
            "conflict_escalation_cost": float(cost_scores[s]),
            "justice_feeling": float(jus_scores[s]),
        }

    # -------------------- Government --------------------
    budget_params = cf.default_budget_cost_gov_params(excel_path=cfg.budget_cost_excel)
    budget_scores = cf.build_budget_cost_scores_gov_for_scenarios(budget_params)

    eg_params = cf.default_economic_growth_gov_params(excel_path=cfg.economic_growth_excel)
    eg_scores = cf.build_economic_growth_scores_gov_for_scenarios(
        eg_params,
        agri_params=agri_params,
        industry_params=ind_params,
        hydro_state=cfg.hydro_state,
    )

    ps_params = cf.default_public_satisfaction_gov_params(excel_path=cfg.public_satisfaction_excel)
    ps_scores = cf.build_public_satisfaction_scores_gov_for_scenarios(ps_params)

    sec_params = cf.default_security_risk_gov_params(excel_path=cfg.security_risk_excel)
    sec_scores = cf.build_security_risk_scores_gov_for_scenarios(sec_params)

    scores_G: Dict[str, Dict[str, float]] = {}
    for s in SCENARIOS:
        scores_G[s] = {
            "budget_cost": float(budget_scores[s]),
            "economic_growth": float(eg_scores[s]),
            "public_satisfaction": float(ps_scores[s]),
            "security_risk": float(sec_scores[s]),
        }

    return scores_I, scores_Y, scores_G


def compute_player_utilities(
    scores_I: Dict[str, Dict[str, float]],
    scores_Y: Dict[str, Dict[str, float]],
    scores_G: Dict[str, Dict[str, float]],
    *,
    normalize_utilities: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compute utilities per player per scenario using AHP weights.

    Returns
    -------
    dict[player] -> dict[scenario] -> utility
    If normalize_utilities=True, utilities are scaled to [-5,+5] per player.
    """
    wI, _ = ahp_weights(pairwise_I)
    wY, _ = ahp_weights(pairwise_Y)
    wG, _ = ahp_weights(pairwise_G)

    weight_I = {criteria_I[i]: float(wI[i]) for i in range(len(criteria_I))}
    weight_Y = {criteria_Y[i]: float(wY[i]) for i in range(len(criteria_Y))}
    weight_G = {criteria_G[i]: float(wG[i]) for i in range(len(criteria_G))}

    utilities_raw = {
        "Isfahan": {s: compute_utility(scores_I[s], weight_I) for s in SCENARIOS},
        "Yazd": {s: compute_utility(scores_Y[s], weight_Y) for s in SCENARIOS},
        "Government": {s: compute_utility(scores_G[s], weight_G) for s in SCENARIOS},
    }

    if not normalize_utilities:
        return utilities_raw

    utilities_scaled: Dict[str, Dict[str, float]] = {}
    for p, u in utilities_raw.items():
        vals = list(u.values())
        scaled = normalize_to_minus5_plus5(vals)
        utilities_scaled[p] = {SCENARIOS[i]: float(scaled[i]) for i in range(len(SCENARIOS))}

    return utilities_scaled


def build_and_show_payoff_tables(
    utilities: Dict[str, Dict[str, float]],
    *,
    order_players: Tuple[str, str, str] = ("Isfahan", "Government", "Yazd"),
) -> pt.PayoffTables:
    """Create and print the two 3x3 payoff tables (one per Yazd strategy)."""
    tables = pt.build_payoff_tables(utilities, order_players=order_players, cell_mode="tuple")
    print(pt.format_payoff_tables_for_print(tables))
    return tables


# =========================
# DEBUG HELPERS (Chapter 6)
# =========================
def _print_ahp_weights():
    wI, _ = ahp_weights(pairwise_I)
    wY, _ = ahp_weights(pairwise_Y)
    wG, _ = ahp_weights(pairwise_G)

    weight_I = {criteria_I[i]: float(wI[i]) for i in range(len(criteria_I))}
    weight_Y = {criteria_Y[i]: float(wY[i]) for i in range(len(criteria_Y))}
    weight_G = {criteria_G[i]: float(wG[i]) for i in range(len(criteria_G))}

    print("\n" + "="*80)
    print("AHP weights (Isfahan):", weight_I)
    print("AHP weights (Yazd):   ", weight_Y)
    print("AHP weights (Gov):    ", weight_G)
    print("="*80 + "\n")

    return weight_I, weight_Y, weight_G


def _range_report(scores: Dict[str, Dict[str, float]], criteria_list, title: str):
    print("\n" + "="*80)
    print(f"RANGE REPORT — {title}")
    print("="*80)
    for c in criteria_list:
        vals = [float(scores[s][c]) for s in SCENARIOS]
        mn, mx = min(vals), max(vals)
        print(f"{c:>30s} | min={mn:8.3f} | max={mx:8.3f} | range={mx-mn:8.3f}")
    print("="*80 + "\n")


def run_debug(cfg: ScoreBuildersConfig):
    # Safety for missing excel paths
    for attr in ["budget_cost_excel", "economic_growth_excel", "public_satisfaction_excel", "security_risk_excel"]:
        p = getattr(cfg, attr)
        if p is not None and (not os.path.exists(p)):
            print(f"[WARN] Excel not found -> using defaults: {attr}='{p}'")
            setattr(cfg, attr, None)

    scores_I, scores_Y, scores_G = build_all_scores(cfg)

    _print_ahp_weights()

    _range_report(scores_I, criteria_I, "Isfahan criterion scores (0..100)")
    _range_report(scores_Y, criteria_Y, "Yazd criterion scores (0..100)")
    _range_report(scores_G, criteria_G, "Government criterion scores (0..100)")

    utilities_raw = compute_player_utilities(scores_I, scores_Y, scores_G, normalize_utilities=False)
    print("\n" + "#"*80)
    print("PAYOFF TABLES — RAW utilities (no normalization)")
    print("#"*80 + "\n")
    build_and_show_payoff_tables(utilities_raw)

    utilities_scaled = compute_player_utilities(scores_I, scores_Y, scores_G, normalize_utilities=True)
    print("\n" + "#"*80)
    print("PAYOFF TABLES — NORMALIZED utilities [-5,+5] (for comparison)")
    print("#"*80 + "\n")
    build_and_show_payoff_tables(utilities_scaled)


# -------------------------------------------------------------------
# Single main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    spi_df = pd.DataFrame({
        "Year": [1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402],
        "Precipitation_mm": [96.0, 96.0, 71.1, 90.1, 154.2, 141.8, 139.0, 64.7, 135.8, 69.8],
        "SPI_yearly": [-0.291609, -0.291609, -1.028773, -0.466278, 1.431400, 1.064299, 0.981405, -1.218245, 0.886669, -1.067259]
    })
    area_df = pd.DataFrame({
        "crop_year": ["1396-1397", "1397-1398", "1399-1400", "1400-1401", "1402-1403"],
        "Area": [197671, 201639, 199913, 192434, 211452]
    })

    cfg = ScoreBuildersConfig(
        hydro_state="normal",
        spi_df=spi_df,
        area_df=area_df,
        budget_cost_excel="/mnt/data/gov_budget_cost_equal_weights_expert.xlsx",
        economic_growth_excel="/mnt/data/gov_economic_growth_template_1400_with_employment.xlsx",
        public_satisfaction_excel="/mnt/data/gov_public_satisfaction_deltas_proposed.xlsx",
        security_risk_excel="/mnt/data/security_risk_impact_table.xlsx",
    )

    run_debug(cfg)
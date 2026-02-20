# criteria_functions.py
"""
Criteria (score) functions for the Isfahan–Yazd–Government water conflict game.

This module is meant to replace manual, scenario-by-scenario score assignment by using
data-informed indicator functions.

Currently implemented:
  - Isfahan: agricultural_income (proxy: cultivated area × factors)

Author: (your name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

HydroState = Literal["dry", "normal", "wet"]
GovPolicy = Literal["Q_G", "C_G", "R_G"]
IsfAction = Literal["S_I", "M_I", "D_I"]
YazdAction = Literal["S_Y", "A_Y"]

I_STRATS = ("S_I", "M_I", "D_I")
G_STRATS = ("C_G", "Q_G", "R_G")
Y_STRATS = ("S_Y", "A_Y")


# ---------------------------------------------------------------------
# 1) SPI -> HydroState (based on SPI "near normal" definition)
#    normal: -0.99 <= SPI <= 0.99
#    wet:    SPI > 0.99
#    dry:    SPI < -0.99
# ---------------------------------------------------------------------
def classify_hydro_state_spi(spi_value: float) -> HydroState:
    if spi_value > 0.99:
        return "wet"
    if spi_value < -0.99:
        return "dry"
    return "normal"


# ---------------------------------------------------------------------
# 2) Helpers
# ---------------------------------------------------------------------
def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize to [0, 100] using min-max scaling."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=float)
    return 100.0 * (values - vmin) / (vmax - vmin)


def _parse_crop_year_start(crop_year: str) -> int:
    """Convert '1396-1397' -> 1396 (start year)."""
    return int(str(crop_year).split("-")[0])


# ---------------------------------------------------------------------
# 3) Agricultural income (Isfahan) calibration + scoring
# ---------------------------------------------------------------------
@dataclass
class AgriIsfahanParams:
    """
    Calibrated parameters for Isfahan agricultural income proxy model.

    BaseArea:
        Mean cultivated area in NORMAL hydro years (within overlap).
    hydro_factor:
        HydroFactor[state] = mean_precip(state) / mean_precip(normal) (computed over overlap).
    policy_factor:
        Q_G fixed at 1.
        R_G anchored to observed max area: max_area / base_area.
        C_G halfway between Q_G and R_G (mild improvement).
    conflict_factor:
        Disruption factors for Isfahan actions (assumption; use sensitivity analysis later).
    """
    base_area: float
    hydro_factor: Dict[HydroState, float]
    policy_factor: Dict[GovPolicy, float]
    conflict_factor: Dict[IsfAction, float]
    overlap_df: pd.DataFrame


def calibrate_agri_isfahan(
        spi_df: pd.DataFrame,
        area_df: pd.DataFrame,
        *,
        spi_col: str = "SPI_yearly",
        precip_col: str = "Precipitation_mm",
        year_col: str = "Year",
        crop_year_col: str = "crop_year",
        area_col: str = "Area",
        conflict_factor: Dict[IsfAction, float] | None = None,
) -> AgriIsfahanParams:
    """
    Calibrate the proxy model for Isfahan agricultural income using:
      - SPI-based hydro classification (dry/normal/wet)
      - precipitation ratios for HydroFactor
      - observed max cultivated area to anchor PolicyFactor(R_G)

    Parameters
    ----------
    spi_df:
        Must include [Year, Precipitation_mm, SPI_yearly] (or renamed via args).
    area_df:
        Must include [crop_year, Area] where crop_year like '1396-1397'.
    conflict_factor:
        Optional dict. If None, defaults to {"S_I":1.0,"M_I":0.95,"D_I":0.80}.

    Returns
    -------
    AgriIsfahanParams
        Includes calibrated factors and the merged overlap dataframe.
    """
    if conflict_factor is None:
        conflict_factor = {"S_I": 0.95, "M_I": 1.1, "D_I": 1.05}

    # Defensive checks
    for c in [year_col, precip_col, spi_col]:
        if c not in spi_df.columns:
            raise ValueError(f"spi_df must contain column '{c}'")
    for c in [crop_year_col, area_col]:
        if c not in area_df.columns:
            raise ValueError(f"area_df must contain column '{c}'")

    # Copy to avoid side effects
    spi = spi_df[[year_col, precip_col, spi_col]].copy()
    area = area_df[[crop_year_col, area_col]].copy()

    area["Year_start"] = area[crop_year_col].apply(_parse_crop_year_start)
    spi["HydroState"] = spi[spi_col].apply(classify_hydro_state_spi)

    overlap = area.merge(
        spi[[year_col, precip_col, spi_col, "HydroState"]],
        left_on="Year_start",
        right_on=year_col,
        how="left"
    ).drop(columns=[year_col])

    # BaseArea = mean area in NORMAL years within overlap
    base_area = float(overlap.loc[overlap["HydroState"] == "normal", area_col].mean())
    if np.isnan(base_area):
        base_area = float(overlap[area_col].mean())

    # HydroFactor = mean_precip(state) / mean_precip(normal), computed over overlap rows
    precip_by_state = overlap.groupby("HydroState")[precip_col].mean()
    precip_normal = precip_by_state.get("normal", np.nan)
    hydro_factor: Dict[HydroState, float] = {}
    for st in ("dry", "normal", "wet"):
        if pd.notna(precip_by_state.get(st, np.nan)) and pd.notna(precip_normal) and precip_normal != 0:
            hydro_factor[st] = float(precip_by_state[st] / precip_normal)
        else:
            hydro_factor[st] = 1.0

    # PolicyFactor anchored to observed max area
    max_area = float(overlap[area_col].max())
    policy_R = float(max_area / base_area) if base_area != 0 else 1.0
    policy_factor: Dict[GovPolicy, float] = {
        "Q_G": 1.0,
        "C_G": 1.0 + 0.5 * (policy_R - 1.0),
        "R_G": policy_R,
    }

    return AgriIsfahanParams(
        base_area=base_area,
        hydro_factor=hydro_factor,
        policy_factor=policy_factor,
        conflict_factor=conflict_factor,
        overlap_df=overlap
    )


def agri_income_index_isfahan(
        params: AgriIsfahanParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        hydro_state: HydroState,
) -> float:
    """
    Compute the agricultural income proxy index (NOT normalized) for one combination.
    """
    return (
            params.base_area
            * params.policy_factor[gov_policy]
            * params.hydro_factor[hydro_state]
            * params.conflict_factor[isf_action]
    )


def build_agri_scores_isfahan_for_scenarios(
        params: AgriIsfahanParams,
        *,
        hydro_state: HydroState = "normal",
        include_yazd_dimension: bool = True,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Isfahan agricultural_income over the game's scenario naming.

    If include_yazd_dimension=True, outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    If False, outputs 9 keys without Yazd suffix:
      "S_I_C_G", ... "D_I_R_G"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            if include_yazd_dimension:
                for Y in Y_STRATS:
                    scen = f"{I}_{G}_{Y}"
                    idx = agri_income_index_isfahan(params, G, I, hydro_state)
                    keys.append(scen)
                    rows.append(idx)
            else:
                scen = f"{I}_{G}"
                idx = agri_income_index_isfahan(params, G, I, hydro_state)
                keys.append(scen)
                rows.append(idx)

    idx_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(idx_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}


# ---------------------------------------------------------------------
# 4) Social satisfaction (Isfahan) - expert-judgment additive model
# ---------------------------------------------------------------------
@dataclass
class SocialIsfahanParams:
    """
    Expert-judgment parameters for Isfahan social satisfaction.

    We model social satisfaction as a weighted sum of three sub-indicators:

      1) social_contentment: perceived public contentment in Isfahan
      2) perceived_justice: perceived fairness of allocation from Isfahan's viewpoint
      3) crisis_response: intensity of social reactions / tensions in Isfahan

    Each player strategy contributes additively to each sub-indicator.
    Values are elicited from the author's expert judgment due to lack of suitable data.
    (You will document this explicitly in the report.)

    Notes
    -----
    - This is an Isfahan-only criterion: all sub-indicators are measured as perceived in Isfahan.
    - Not every actor needs to affect every sub-indicator; use 0 where no direct effect is assumed.
    - Weights can later be set via AHP; defaults are equal weights.
    """
    contentment_effect: Dict[str, float]
    justice_effect: Dict[str, float]
    crisis_effect: Dict[str, float]
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)


def default_social_isfahan_params() -> SocialIsfahanParams:
    """
    Return the default expert-judgment effects table (as validated by the author).

    Effects are defined for:
      - Isfahan actions: S_I, M_I, D_I
      - Yazd actions:    S_Y, A_Y
      - Government pol.: Q_G, C_G, R_G
    """
    # Contentment effects (Isfahan viewpoint)
    contentment = {
        "S_I": 0,
        "M_I": 7,
        "D_I": 5,
        "S_Y": 0,
        "A_Y": -4,
        "Q_G": -6,
        "C_G": 3,
        "R_G": 5,
    }

    # Perceived justice effects (Isfahan viewpoint)
    justice = {
        "S_I": 0,
        "M_I": 0,
        "D_I": 0,
        "S_Y": 0,
        "A_Y": 0,
        "Q_G": 0,
        "C_G": 2,
        "R_G": 5,
    }

    # Crisis-response effects (Isfahan viewpoint)
    # -- Crisis-response here represents social unrest intensity in Isfahan
    crisis = {
        "S_I": 0,
        "M_I": -6,
        "D_I": -4,
        "S_Y": 0,
        "A_Y": -2,
        "Q_G": -5,
        "C_G": 4,
        "R_G": 5,
    }

    return SocialIsfahanParams(
        contentment_effect=contentment,
        justice_effect=justice,
        crisis_effect=crisis,
        weights=(1/3, 1/3, 1/3),
    )


def social_satisfaction_index_isfahan(
        params: SocialIsfahanParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
) -> float:
    """
    Compute Isfahan social satisfaction index (NOT normalized) for one scenario.

    The index is computed as:
      w1 * (sum of contentment effects)
    + w2 * (sum of perceived justice effects)
    + w3 * (sum of crisis-response effects)

    where each component is the additive sum of contributions from:
      - Isfahan action
      - Yazd action
      - Government policy
    """
    w1, w2, w3 = params.weights

    contentment = (
            params.contentment_effect.get(isf_action, 0.0)
            + params.contentment_effect.get(yazd_action, 0.0)
            + params.contentment_effect.get(gov_policy, 0.0)
    )

    justice = (
            params.justice_effect.get(isf_action, 0.0)
            + params.justice_effect.get(yazd_action, 0.0)
            + params.justice_effect.get(gov_policy, 0.0)
    )

    crisis = (
            params.crisis_effect.get(isf_action, 0.0)
            + params.crisis_effect.get(yazd_action, 0.0)
            + params.crisis_effect.get(gov_policy, 0.0)
    )

    return float(w1 * contentment + w2 * justice + w3 * crisis)


def build_social_scores_isfahan_for_scenarios(
        params: SocialIsfahanParams,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Isfahan social_satisfaction over the game's scenario naming.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                idx = social_satisfaction_index_isfahan(params, G, I, Y)
                keys.append(scen)
                rows.append(idx)

    idx_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(idx_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}



# ---------------------------------------------------------------------
# 5) Environmental status (Isfahan) - expert-judgment additive model
# ---------------------------------------------------------------------
@dataclass
class EnvIsfahanParams:
    """
    Expert-judgment parameters for Isfahan environmental/eco status.

    We model environmental status as a weighted sum of three sub-indicators:

      1) eflow: environmental flow condition of Zayandeh-Rood (reference: 10 m^3/s)
      2) wetland: ecological condition of Gavkhouni wetland (reference: environmental water right)
      3) sustainability: long-term ecological sustainability of the basin

    Each player strategy contributes additively to each sub-indicator.
    Values are elicited from the author's expert judgment due to lack of suitable data.
    (Documented explicitly in the report.)

    Notes
    -----
    - This is an Isfahan-only criterion: all sub-indicators are assessed from Isfahan's viewpoint.
    - Not every actor must affect every sub-indicator; use 0 where no direct effect is assumed.
    - Weights are set equal here (as confirmed by the author).
    """
    eflow_effect: Dict[str, float]
    wetland_effect: Dict[str, float]
    sustainability_effect: Dict[str, float]
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)


def default_env_isfahan_params() -> EnvIsfahanParams:
    """
    Return the default expert-judgment effects table (as validated by the author).

    Effects are defined for:
      - Isfahan actions: S_I, M_I, D_I
      - Yazd actions:    S_Y, A_Y
      - Government pol.: Q_G, C_G, R_G
    """
    # Environmental flow effects (Zayandeh-Rood; reference: 10 m^3/s)
    eflow = {
        "S_I": 0,
        "M_I": 4,
        "D_I": 2,
        "S_Y": 0,
        "A_Y": -4,
        "Q_G": -4,
        "C_G": 0,
        "R_G": 5,
    }

    # Gavkhouni wetland condition effects (reference: environmental water right)
    wetland = {
        "S_I": 0,
        "M_I": 3,
        "D_I": 1,
        "S_Y": 0,
        "A_Y": -3,
        "Q_G": -3,
        "C_G": 0,
        "R_G": 4,
    }

    # Long-term ecological sustainability effects
    sustainability = {
        "S_I": 0,
        "M_I": 3,
        "D_I": 2,
        "S_Y": 0,
        "A_Y": -2,
        "Q_G": -4,
        "C_G": 0,
        "R_G": 3,
    }

    return EnvIsfahanParams(
        eflow_effect=eflow,
        wetland_effect=wetland,
        sustainability_effect=sustainability,
        weights=(1/3, 1/3, 1/3),
    )


def env_status_index_isfahan(
        params: EnvIsfahanParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
) -> float:
    """
    Compute Isfahan environmental status index (NOT normalized) for one scenario.

    The index is computed as:
      w1 * (sum of eflow effects)
    + w2 * (sum of wetland effects)
    + w3 * (sum of sustainability effects)

    where each component is the additive sum of contributions from:
      - Isfahan action
      - Yazd action
      - Government policy
    """
    w1, w2, w3 = params.weights

    eflow = (
            params.eflow_effect.get(isf_action, 0.0)
            + params.eflow_effect.get(yazd_action, 0.0)
            + params.eflow_effect.get(gov_policy, 0.0)
    )

    wetland = (
            params.wetland_effect.get(isf_action, 0.0)
            + params.wetland_effect.get(yazd_action, 0.0)
            + params.wetland_effect.get(gov_policy, 0.0)
    )

    sustainability = (
            params.sustainability_effect.get(isf_action, 0.0)
            + params.sustainability_effect.get(yazd_action, 0.0)
            + params.sustainability_effect.get(gov_policy, 0.0)
    )

    return float(w1 * eflow + w2 * wetland + w3 * sustainability)


def build_env_scores_isfahan_for_scenarios(
        params: EnvIsfahanParams,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Isfahan environmental_status over the game's scenario naming.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                idx = env_status_index_isfahan(params, G, I, Y)
                keys.append(scen)
                rows.append(idx)

    idx_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(idx_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}




# ---------------------------------------------------------------------
# 6) Drinking water security (Yazd) - semi data-informed multiplicative model
# ---------------------------------------------------------------------
@dataclass
class WaterSecurityYazdParams:
    """
    Semi data-informed parameters for Yazd drinking-water security.

    We model the drinking-water security as a supply-to-demand ratio:

        index = (annual_supply_mcm * strategy_factor * precip_factor) / annual_demand_mcm

    where:
      - annual_demand_mcm is estimated from population × per-capita drinking-water use
      - annual_supply_mcm is approximated from reported produced/available water in the distribution system
      - strategy_factor captures policy/behavioral impacts on the effective *available* supply (additive effects
        were used for some Isfahan criteria; here we use a multiplicative availability factor, consistent with the
        idea of disruptions or reallocations changing delivered volume)
      - precip_factor captures the (small) environmental effect of wet/dry years on overall availability.
        It is *independent* of strategies.

    Notes
    -----
    - This is a Yazd-only criterion: the score represents Yazd's welfare/utility about drinking-water security.
    - Due to limited public access to official time-series on withdrawals/production, the model uses a mix of
      reported operational figures and expert judgment for strategy impacts. Document this explicitly in the thesis.
    - The hydro_state parameter is used only to apply a small precipitation adjustment (default: 'normal' for
      multi-year averages).
    """
    # Demand side
    population_served: float                      # persons
    per_capita_lpd: float                         # liters per person per day

    # Supply side
    annual_supply_mcm_base: float                 # million m^3 per year (baseline)

    # Strategy impacts (multiplicative factors on supply)
    yazd_factor: Dict[YazdAction, float]
    gov_factor: Dict[GovPolicy, float]
    isf_factor: Dict[IsfAction, float]

    # Small precipitation adjustment (independent of strategies)
    precip_factor: Dict[HydroState, float] = None


def default_water_security_yazd_params() -> WaterSecurityYazdParams:
    """
    Default parameterization for Yazd drinking-water security.

    Data anchors (to be cited in the report):
      - Per-capita household water use in Yazd cities ~159 L/person/day (IRNA report).
      - Operational supply (example): 115 million m^3 in 9 months -> ~145 million m^3/year (Mehr report).
      - Long-term mean annual precipitation in Yazd ~96.4 mm (Shargh report; used only qualitatively here).

    Strategy factors:
      - Yazd active pursuit increases effective supply (better allocation/operations): +10% (1.10)
      - Gov reallocation (R_G) reduces Yazd share (to appease Isfahan): -15% (0.85)
      - Isfahan protest (M_I) and disruptive action (D_I) reduce delivered supply due to tension/disruptions:
        -5% and -10% respectively.

    These strategy factors are expert-judgment parameters and can be refined via sensitivity analysis.
    """
    # Demand side (baseline): population served as cited in your report; per-capita from IRNA.
    population_served = 1_470_000
    per_capita_lpd = 159.0

    # Supply side (baseline): derived from Mehr's "115 million m^3 in 9 months" -> ~145 mcm/year.
    annual_supply_mcm_base = 145.0

    yazd_factor = {
        "S_Y": 1.0,
        "A_Y": 1.10,
    }

    gov_factor = {
        "Q_G": 1.0,   # status quo
        "C_G": 1.0,   # water-right compensation: assumed neutral for drinking-water volume
        "R_G": 0.85,  # reallocation reduces Yazd's share
    }

    isf_factor = {
        "S_I": 1.0,
        "M_I": 0.95,
        "D_I": 0.90,
    }

    # Small precip effect (independent of strategies). For multi-year averages use 'normal' -> 1.0.
    precip_factor = {
        "dry": 0.95,
        "normal": 1.0,
        "wet": 1.05,
    }

    return WaterSecurityYazdParams(
        population_served=population_served,
        per_capita_lpd=per_capita_lpd,
        annual_supply_mcm_base=annual_supply_mcm_base,
        yazd_factor=yazd_factor,
        gov_factor=gov_factor,
        isf_factor=isf_factor,
        precip_factor=precip_factor,
    )


def annual_demand_mcm_yazd(params: WaterSecurityYazdParams) -> float:
    """Compute annual drinking-water demand in million m^3/year from population and per-capita use."""
    # liters/day -> m^3/day (divide by 1000), then annual, then /1e6 to million m^3
    demand_mcm = params.population_served * (params.per_capita_lpd / 1000.0) * 365.0 / 1_000_000.0
    return float(demand_mcm)


def water_security_index_yazd(
        params: WaterSecurityYazdParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
        *,
        hydro_state: HydroState = "normal",
) -> float:
    """
    Compute Yazd drinking-water security index (NOT normalized) for one scenario.

    index = (effective_supply / demand)

    effective_supply = base_supply
                     * gov_factor[gov_policy]
                     * isf_factor[isf_action]
                     * yazd_factor[yazd_action]
                     * precip_factor[hydro_state]
    """
    demand = annual_demand_mcm_yazd(params)

    supply = (
        params.annual_supply_mcm_base
        * params.gov_factor.get(gov_policy, 1.0)
        * params.isf_factor.get(isf_action, 1.0)
        * params.yazd_factor.get(yazd_action, 1.0)
        * (params.precip_factor.get(hydro_state, 1.0) if params.precip_factor else 1.0)
    )

    if demand <= 0:
        return 0.0
    return float(supply / demand)


def build_water_security_scores_yazd_for_scenarios(
        params: WaterSecurityYazdParams,
        *,
        hydro_state: HydroState = "normal",
) -> Dict[str, float]:
    """
    Produce a score dictionary for Yazd water_security over the game's scenario naming.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                idx = water_security_index_yazd(
                    params,
                    gov_policy=G,
                    isf_action=I,
                    yazd_action=Y,
                    hydro_state=hydro_state,
                )
                keys.append(scen)
                rows.append(idx)

    idx_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(idx_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}





# ---------------------------------------------------------------------
# 7) Industrial sustainability (Yazd) - supply/need ratio with additive strategy deltas
# ---------------------------------------------------------------------
@dataclass
class IndustryYazdParams:
    """
    Semi data-informed parameters for Yazd industrial activity sustainability.

    Based on the report's modeling choice (Section 5.5), industrial sustainability is
    modeled as a supply-to-need ratio where strategies change the effective *available*
    industrial water supply:

        U_ind = (Supply0_ind * (1 + ΔY(s_Y) + ΔG(s_G) + ΔI(s_I))) / Need_ind

    Notes
    -----
    - This is a Yazd-only criterion: it reflects Yazd's welfare regarding industrial water sustainability.
    - Supply0_ind and Need_ind are data-anchored where available (and otherwise expert judgment, as documented).
    - Strategy deltas are expressed as fractions (e.g., +0.10 means +10% on Supply0_ind).
    """
    supply0_ind_mcm: float                 # million m^3 per year (baseline industrial supply)
    need_ind_mcm: float                    # million m^3 per year (industrial need)

    yazd_delta: Dict[YazdAction, float]    # ΔY
    gov_delta: Dict[GovPolicy, float]      # ΔG
    isf_delta: Dict[IsfAction, float]      # ΔI


def default_industry_yazd_params() -> IndustryYazdParams:
    """
    Default parameterization for Yazd industrial sustainability.

    Data anchors (as in the report draft):
      - Baseline industrial supply Supply0_ind ≈ 65 MCM/year.
      - Industrial need Need_ind ≈ 90 MCM/year (expert judgment).

    Strategy deltas (initial proposal; can be revised after your confirmation):
      - Yazd active pursuit (A_Y): +10%  -> ΔY = +0.10
      - Government reallocation (R_G): -12% -> ΔG = -0.12
      - Government compensation (C_G): 0% -> assumed budgetary, not directly changing volumes
      - Isfahan moderate protest (M_I): -10% -> ΔI = -0.10
      - Isfahan disruptive action (D_I): -15% -> ΔI = -0.15
    """
    supply0_ind_mcm = 65.0
    need_ind_mcm = 90.0

    yazd_delta = {
        "S_Y": 0.0,
        "A_Y": 0.10,
    }

    gov_delta = {
        "Q_G": 0.0,
        "C_G": 0.0,
        "R_G": -0.12,
    }

    isf_delta = {
        "S_I": 0.0,
        "M_I": -0.10,
        "D_I": -0.15,
    }

    return IndustryYazdParams(
        supply0_ind_mcm=supply0_ind_mcm,
        need_ind_mcm=need_ind_mcm,
        yazd_delta=yazd_delta,
        gov_delta=gov_delta,
        isf_delta=isf_delta,
    )


def industry_sustainability_index_yazd(
        params: IndustryYazdParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
) -> float:
    """
    Compute Yazd industrial sustainability index (NOT normalized) for one scenario.

    U_ind = (Supply0_ind * (1 + ΔY + ΔG + ΔI)) / Need_ind
    """
    need = params.need_ind_mcm
    if need <= 0:
        return 0.0

    delta = (
        params.yazd_delta.get(yazd_action, 0.0)
        + params.gov_delta.get(gov_policy, 0.0)
        + params.isf_delta.get(isf_action, 0.0)
    )

    supply_eff = params.supply0_ind_mcm * (1.0 + delta)

    # Prevent negative effective supply in extreme hypothetical cases.
    supply_eff = max(0.0, float(supply_eff))

    return float(supply_eff / need)


def build_industry_sustainability_scores_yazd_for_scenarios(
        params: IndustryYazdParams,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Yazd industrial_sustainability over the game's scenario naming.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                idx = industry_sustainability_index_yazd(
                    params,
                    gov_policy=G,
                    isf_action=I,
                    yazd_action=Y,
                )
                keys.append(scen)
                rows.append(idx)

    idx_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(idx_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}

# ---------------------------------------------------------------------
# 8) Perceived justice (Yazd) - expert-judgment additive model
# ---------------------------------------------------------------------
@dataclass
class JusticeYazdParams:
    """
    Expert-judgment parameters for Yazd perceived justice (fairness of water allocation).

    Based on the report's modeling choice (Section 5.6), perceived justice is modeled as
    an additive index around a neutral baseline:

        J = J0 + Δ_Y(s_Y) + Δ_G(s_G) + Δ_I(s_I)

    where J0 is set to 0 (neutral reference), and each actor's strategy contributes an
    additive effect to Yazd's perceived justice.

    Notes
    -----
    - This is a Yazd-only criterion: all values are measured as perceived in Yazd.
    - Effects are elicited from the author's expert judgment when direct quantitative
      data is not available; the effects table can be updated and sensitivity-tested.
    """
    yazd_effect: Dict[YazdAction, float]
    gov_effect: Dict[GovPolicy, float]
    isf_effect: Dict[IsfAction, float]
    base: float = 0.0


def default_justice_yazd_params() -> JusticeYazdParams:
    """
    Default expert-judgment effects for Yazd perceived justice.

    The default values are taken from the provided impact table (Excel), consistent with
    the narrative assumptions in the report (Section 5.6):
      - Yazd active pursuit slightly increases perceived justice (recognition/agency).
      - Government reallocation to Isfahan decreases Yazd perceived justice.
      - Isfahan protest/disruption decreases Yazd perceived justice.

    You can revise these deltas later if you elicit expert opinion or new evidence.
    """
    yazd_effect = {
        "S_Y": 0.0,
        "A_Y": 4.0,
    }

    gov_effect = {
        "Q_G": 0.0,
        "C_G": 0.0,
        "R_G": -6.0,
    }

    isf_effect = {
        "S_I": 0.0,
        "M_I": -3.0,
        "D_I": -5.0,
    }

    return JusticeYazdParams(
        yazd_effect=yazd_effect,
        gov_effect=gov_effect,
        isf_effect=isf_effect,
        base=0.0,
    )


def perceived_justice_index_yazd(
        params: JusticeYazdParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
) -> float:
    """
    Compute Yazd perceived justice index (NOT normalized) for one scenario.

    J = base + Δ_Y(s_Y) + Δ_G(s_G) + Δ_I(s_I)
    """
    return float(
        params.base
        + params.yazd_effect.get(yazd_action, 0.0)
        + params.gov_effect.get(gov_policy, 0.0)
        + params.isf_effect.get(isf_action, 0.0)
    )


def build_perceived_justice_scores_yazd_for_scenarios(
        params: JusticeYazdParams,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Yazd perceived_justice over the game's scenario naming.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                idx = perceived_justice_index_yazd(params, G, I, Y)
                keys.append(scen)
                rows.append(idx)

    idx_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(idx_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}





# ---------------------------------------------------------------------
# 9) Budget cost (Government) - expert-judgment additive *cost* model
# ---------------------------------------------------------------------
@dataclass
class BudgetCostGovParams:
    """
    Expert-judgment parameters for Government budget cost.

    We model Government budget cost as a *cost* indicator (higher = worse), constructed from
    three sub-indicators (0..10 scale each):

      B1: Policy/Admin costs
      B2: Compensation/Transfers costs
      B3: Security/Repair (crisis management) costs

    For each sub-indicator, each actor's strategy contributes additively:

        Bk = Δk_I(s_I) + Δk_G(s_G) + Δk_Y(s_Y)   for k in {1,2,3}

    The overall (raw) budget cost index is the equal-weight average:

        BC = (B1 + B2 + B3) / 3

    Notes
    -----
    - This is a Government-only criterion (higher cost is worse for Government).
    - Due to lack of scenario-level budget data, effects are elicited via structured expert judgment
      using a 0..10 scale (documented in the thesis).
    - In the utility aggregation, LOWER cost should map to HIGHER utility. Therefore, scenario
      scores returned by `build_budget_cost_scores_gov_for_scenarios` are direction-reversed
      (min-max on cost, then 100 - normalized cost).
    """
    # Per-actor, per-strategy effects for each sub-indicator (0..10)
    b1_effect_isf: Dict[IsfAction, float]
    b1_effect_gov: Dict[GovPolicy, float]
    b1_effect_yazd: Dict[YazdAction, float]

    b2_effect_isf: Dict[IsfAction, float]
    b2_effect_gov: Dict[GovPolicy, float]
    b2_effect_yazd: Dict[YazdAction, float]

    b3_effect_isf: Dict[IsfAction, float]
    b3_effect_gov: Dict[GovPolicy, float]
    b3_effect_yazd: Dict[YazdAction, float]

    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)


def default_budget_cost_gov_params(
        *,
        excel_path: str | None = None,
) -> BudgetCostGovParams:
    """
    Load default effects for Government budget cost.

    If excel_path is provided, expects an Excel file with a sheet (default: 'Strategy_Effects')
    containing columns:

      - Actor
      - Strategy
      - Effect_on_B1 (0..10)
      - Effect_on_B2 (0..10)
      - Effect_on_B3 (0..10)

    Actors should be one of: Esfahan, Yazd, Government (case-insensitive).

    If excel_path is None or reading fails, uses safe defaults consistent with the
    agreed equal-weights expert table.

    This keeps the code runnable even if the Excel file is not present.
    """
    # Safe defaults (equal-weights expert judgment; can be revised by replacing via Excel)
    b1_gov = {"Q_G": 2.0, "C_G": 4.0, "R_G": 7.0}
    b2_gov = {"Q_G": 0.0, "C_G": 8.0, "R_G": 2.0}
    b3_gov = {"Q_G": 3.0, "C_G": 2.0, "R_G": 5.0}

    b1_isf = {"S_I": 0.0, "M_I": 0.0, "D_I": 0.0}
    b2_isf = {"S_I": 0.0, "M_I": 0.0, "D_I": 0.0}
    b3_isf = {"S_I": 0.0, "M_I": 4.0, "D_I": 8.0}

    b1_yazd = {"S_Y": 0.0, "A_Y": 2.0}
    b2_yazd = {"S_Y": 0.0, "A_Y": 0.0}
    b3_yazd = {"S_Y": 0.0, "A_Y": 0.0}

    if excel_path:
        try:
            df = pd.read_excel(excel_path, sheet_name="Strategy_Effects")
            for _, row in df.iterrows():
                actor = str(row.get("Actor", "")).strip().lower()
                strat = str(row.get("Strategy", "")).strip()
                b1 = row.get("Effect_on_B1 (0..10)", 0.0)
                b2 = row.get("Effect_on_B2 (0..10)", 0.0)
                b3 = row.get("Effect_on_B3 (0..10)", 0.0)

                def _to_float(x):
                    return 0.0 if (pd.isna(x) or x == "") else float(x)

                b1, b2, b3 = _to_float(b1), _to_float(b2), _to_float(b3)

                if actor == "government" and strat:
                    b1_gov[strat] = b1
                    b2_gov[strat] = b2
                    b3_gov[strat] = b3
                elif actor == "esfahan" and strat:
                    b1_isf[strat] = b1
                    b2_isf[strat] = b2
                    b3_isf[strat] = b3
                elif actor == "yazd" and strat:
                    b1_yazd[strat] = b1
                    b2_yazd[strat] = b2
                    b3_yazd[strat] = b3
        except Exception:
            # Fall back to defaults silently (documented in report)
            pass

    # Ensure required keys exist
    for k in I_STRATS:
        b1_isf.setdefault(k, 0.0); b2_isf.setdefault(k, 0.0); b3_isf.setdefault(k, 0.0)
    for k in Y_STRATS:
        b1_yazd.setdefault(k, 0.0); b2_yazd.setdefault(k, 0.0); b3_yazd.setdefault(k, 0.0)
    for k in ("Q_G", "C_G", "R_G"):
        b1_gov.setdefault(k, 0.0); b2_gov.setdefault(k, 0.0); b3_gov.setdefault(k, 0.0)

    return BudgetCostGovParams(
        b1_effect_isf=b1_isf, b1_effect_gov=b1_gov, b1_effect_yazd=b1_yazd,
        b2_effect_isf=b2_isf, b2_effect_gov=b2_gov, b2_effect_yazd=b2_yazd,
        b3_effect_isf=b3_isf, b3_effect_gov=b3_gov, b3_effect_yazd=b3_yazd,
        weights=(1/3, 1/3, 1/3),
    )


def budget_cost_index_gov(
        params: BudgetCostGovParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
) -> float:
    """
    Compute Government budget *cost* index (NOT normalized) for one scenario.

        B1 = ΔB1_I + ΔB1_G + ΔB1_Y
        B2 = ΔB2_I + ΔB2_G + ΔB2_Y
        B3 = ΔB3_I + ΔB3_G + ΔB3_Y
        BC = (B1 + B2 + B3) / 3
    """
    w1, w2, w3 = params.weights

    b1 = (
        params.b1_effect_isf.get(isf_action, 0.0)
        + params.b1_effect_gov.get(gov_policy, 0.0)
        + params.b1_effect_yazd.get(yazd_action, 0.0)
    )
    b2 = (
        params.b2_effect_isf.get(isf_action, 0.0)
        + params.b2_effect_gov.get(gov_policy, 0.0)
        + params.b2_effect_yazd.get(yazd_action, 0.0)
    )
    b3 = (
        params.b3_effect_isf.get(isf_action, 0.0)
        + params.b3_effect_gov.get(gov_policy, 0.0)
        + params.b3_effect_yazd.get(yazd_action, 0.0)
    )

    return float(w1 * b1 + w2 * b2 + w3 * b3)


def build_budget_cost_scores_gov_for_scenarios(
        params: BudgetCostGovParams,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Government budget cost over the game's scenario naming.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
        Higher score means LOWER cost (better for Government).
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                cost = budget_cost_index_gov(params, G, I, Y)
                keys.append(scen)
                rows.append(cost)

    cost_arr = np.array(rows, dtype=float)
    score_arr = _min_max_normalize_inverse(cost_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}




# ---------------------------------------------------------------------
# 10) Economic growth & employment (Government) - data-informed composite model
# ---------------------------------------------------------------------
@dataclass
class EconomicGrowthGovParams:
    """
    Data-informed parameters for Government economic growth / economic performance (Esfahan+Yazd focus).

    We construct a composite index from three sub-indicators:

      E1(s): Isfahan agricultural economic score (already computed as [0,100])
      E2(s): Yazd industrial sustainability score (already computed as [0,100])
      E3(s): Employment proxy, built from E1 and E2 and calibrated by provincial unemployment rates:

          E3_raw(s) = 0.5 * ((1-u_isf)*E1(s) + (1-u_yazd)*E2(s))

    The raw economic-growth index is:

          EG_raw(s) = w1*E1(s) + w2*E2(s) + w3*E3_raw(s)

    where weights are data-informed:
      - w3 is set by agreement (default 0.30)
      - w1 and w2 are scaled to sum (1-w3) using value-added shares (year 1400):
            w1 ∝ A_share_isf
            w2 ∝ I_share_yazd

    Notes
    -----
    - This is a Government-only criterion: higher EG means better economic performance.
    - Strategy effects enter indirectly through E1 and E2 (consistent with the report's logic).
    - Final scenario scores are normalized to [0,100] using min-max on EG_raw across 18 scenarios.
    """
    # Data anchors (year 1400)
    a_share_isf_agri_pct: float = 4.2   # Isfahan share of national VA in agriculture (%)
    i_share_yazd_ind_pct: float = 3.4   # Yazd share of national VA in industry (%)

    # Labor-market calibration (Autumn 1400)
    u_isf: float = 0.114  # unemployment rate (fraction, not percent)
    u_yazd: float = 0.105

    # Weighting
    w3_employment: float = 0.30
    # If provided, w1/w2 overrides can be used (rare); keep None by default.
    w1_override: float | None = None
    w2_override: float | None = None

    def weights(self) -> Tuple[float, float, float]:
        """Return (w1,w2,w3) that sum to 1.0."""
        w3 = float(self.w3_employment)
        if self.w1_override is not None and self.w2_override is not None:
            w1 = float(self.w1_override)
            w2 = float(self.w2_override)
            # Normalize if user-provided overrides do not sum to (1-w3)
            s = w1 + w2
            if s > 0:
                w1 = (1.0 - w3) * w1 / s
                w2 = (1.0 - w3) * w2 / s
            return w1, w2, w3

        denom = float(self.a_share_isf_agri_pct + self.i_share_yazd_ind_pct)
        if denom <= 0:
            # Fallback to equal split between E1 and E2
            w1 = (1.0 - w3) * 0.5
            w2 = (1.0 - w3) * 0.5
        else:
            w1 = (1.0 - w3) * float(self.a_share_isf_agri_pct) / denom
            w2 = (1.0 - w3) * float(self.i_share_yazd_ind_pct) / denom
        return w1, w2, w3


def default_economic_growth_gov_params(
        *,
        excel_path: str | None = None,
) -> EconomicGrowthGovParams:
    """
    Load default parameters for Government economic growth.

    If excel_path is provided, attempts to read 'Data_1400' from the template Excel and
    extract:
      - A_share (Isfahan agri VA share, %)
      - I_share (Yazd industry VA share, %)
      - Isfahan/Yazd unemployment rates (percent -> fraction)

    If reading fails or excel_path is None, uses safe defaults:
      A_share=4.2, I_share=3.4, u_isf=0.114, u_yazd=0.105, w3=0.30

    This keeps the code runnable even if the Excel file is missing.
    """
    params = EconomicGrowthGovParams()

    if excel_path:
        try:
            df = pd.read_excel(excel_path, sheet_name="Data_1400")
            # Expect columns: Field, Value, Unit, Source (short)
            field_to_value = {str(r["Field"]).strip(): r["Value"] for _, r in df.iterrows() if "Field" in r}

            # Shares (percent)
            a = field_to_value.get("Isfahan share of national VA in Agriculture (A_share)", params.a_share_isf_agri_pct)
            i = field_to_value.get("Yazd share of national VA in Industry (I_share)", params.i_share_yazd_ind_pct)
            # Unemployment (percent -> fraction)
            u_isf_pct = field_to_value.get("Isfahan unemployment rate (Autumn 1400)", params.u_isf * 100.0)
            u_yazd_pct = field_to_value.get("Yazd unemployment rate (Autumn 1400)", params.u_yazd * 100.0)

            def _to_float(x, default):
                try:
                    if pd.isna(x) or x == "":
                        return float(default)
                    return float(x)
                except Exception:
                    return float(default)

            params.a_share_isf_agri_pct = _to_float(a, params.a_share_isf_agri_pct)
            params.i_share_yazd_ind_pct = _to_float(i, params.i_share_yazd_ind_pct)
            params.u_isf = _to_float(u_isf_pct, params.u_isf * 100.0) / 100.0
            params.u_yazd = _to_float(u_yazd_pct, params.u_yazd * 100.0) / 100.0
        except Exception:
            pass

    return params


def economic_growth_index_gov_raw(
        params: EconomicGrowthGovParams,
        e1_score: float,
        e2_score: float,
) -> float:
    """Compute EG_raw for one scenario given E1 and E2 (both expected in [0,100])."""
    w1, w2, w3 = params.weights()
    e3_raw = 0.5 * ((1.0 - params.u_isf) * float(e1_score) + (1.0 - params.u_yazd) * float(e2_score))
    return float(w1 * float(e1_score) + w2 * float(e2_score) + w3 * e3_raw)


def build_economic_growth_scores_gov_for_scenarios(
        params: EconomicGrowthGovParams,
        *,
        agri_params: AgriIsfahanParams,
        industry_params: IndustryYazdParams,
        hydro_state: HydroState = "normal",
) -> Dict[str, float]:
    """
    Build Government economic-growth scores over the game's 18 scenarios.

    This function is *data-informed* and fully consistent with the report's logic:
      - E1(s): computed by Isfahan agri score builder (normalized [0,100])
      - E2(s): computed by Yazd industry score builder (normalized [0,100])
      - E3(s): employment proxy from E1 and E2, calibrated by unemployment rates
      - EG_raw(s): weighted sum (w3 fixed by agreement; w1/w2 from VA shares)

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
    """
    e1_scores = build_agri_scores_isfahan_for_scenarios(
        agri_params,
        hydro_state=hydro_state,
        include_yazd_dimension=True,
    )
    e2_scores = build_industry_sustainability_scores_yazd_for_scenarios(industry_params)

    keys = []
    rows = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                e1 = float(e1_scores.get(scen, 0.0))
                e2 = float(e2_scores.get(scen, 0.0))
                eg = economic_growth_index_gov_raw(params, e1, e2)
                keys.append(scen)
                rows.append(eg)

    eg_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(eg_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}


# ---------------------------------------------------------------------
# 11) Public satisfaction & social stability (Government) - expert-judgment additive model
# ---------------------------------------------------------------------
@dataclass
class PublicSatisfactionGovParams:
    """
    Expert-judgment parameters for Government public satisfaction & social stability.

    This criterion is a *benefit* indicator (higher = better for Government). It is built from
    three sub-indicators (raw additive scales):

      S1: Perceived satisfaction in Isfahan (higher = more satisfied)
      S2: Perceived satisfaction in Yazd    (higher = more satisfied)
      S3: Social stability (higher = more stability / less protest)

    Each sub-indicator is modeled as an additive sum of independent strategy effects:

        S_k = Δk_I(s_I) + Δk_G(s_G) + Δk_Y(s_Y)   for k in {1,2,3}

    The overall raw index is a weighted sum:

        PS_raw = w1*S1 + w2*S2 + w3*S3

    Due to the lack of suitable quantitative scenario-level data for social satisfaction,
    the effects are elicited via structured expert judgment, consistent with the report.

    Final scenario scores are normalized to [0,100] using min-max across 18 scenarios.
    """
    s1_effect: Dict[str, float]
    s2_effect: Dict[str, float]
    s3_effect: Dict[str, float]
    weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)


def default_public_satisfaction_gov_params(
        *,
        excel_path: str | None = None,
) -> PublicSatisfactionGovParams:
    """
    Load default effects for Government public satisfaction & social stability.

    If excel_path is provided, expects an Excel file with a sheet named 'Deltas_by_strategy'
    containing columns:
      - Sub-index
      - Actor
      - Strategy
      - Delta (raw)

    Sub-index values should be:
      - S1_Isf_satisfaction
      - S2_Yazd_satisfaction
      - S3_Social_stability

    If reading fails or excel_path is None, uses the agreed expert-judgment defaults.
    """
    # Agreed defaults (as validated by the author in the final Excel table)
    s1 = {
        "Q_G": -0.5, "C_G": 0.8, "R_G": 1.2,
        "S_I": 0.0, "M_I": -0.3, "D_I": -0.8,
        "S_Y": 0.0, "A_Y": -0.2,
    }
    s2 = {
        "Q_G": -0.3, "C_G": 0.4, "R_G": -1.0,
        "S_Y": -0.1, "A_Y": 0.3,
        "S_I": 0.0, "M_I": -0.4, "D_I": -1.0,
    }
    s3 = {
        "S_I": 0.2, "M_I": -0.8, "D_I": -1.5,
        "S_Y": 0.1, "A_Y": -0.2,
        "Q_G": -0.4, "C_G": 0.6, "R_G": 0.2,
    }

    if excel_path:
        try:
            df = pd.read_excel(excel_path, sheet_name="Deltas_by_strategy")
            # Defensive parsing
            for _, row in df.iterrows():
                sub = str(row.get("Sub-index", "")).strip()
                strat = str(row.get("Strategy", "")).strip()
                delta = row.get("Delta (raw)", 0.0)
                delta = 0.0 if (pd.isna(delta) or delta == "") else float(delta)
                if not strat:
                    continue
                if sub == "S1_Isf_satisfaction":
                    s1[strat] = delta
                elif sub == "S2_Yazd_satisfaction":
                    s2[strat] = delta
                elif sub == "S3_Social_stability":
                    s3[strat] = delta
        except Exception:
            # Fall back to defaults silently (documented in report)
            pass

    # Ensure required keys exist
    for k in I_STRATS:
        s1.setdefault(k, 0.0); s2.setdefault(k, 0.0); s3.setdefault(k, 0.0)
    for k in Y_STRATS:
        s1.setdefault(k, 0.0); s2.setdefault(k, 0.0); s3.setdefault(k, 0.0)
    for k in ("Q_G", "C_G", "R_G"):
        s1.setdefault(k, 0.0); s2.setdefault(k, 0.0); s3.setdefault(k, 0.0)

    return PublicSatisfactionGovParams(
        s1_effect=s1,
        s2_effect=s2,
        s3_effect=s3,
        weights=(1/3, 1/3, 1/3),
    )


def public_satisfaction_index_gov_raw(
        params: PublicSatisfactionGovParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
) -> float:
    """
    Compute Government public-satisfaction raw index for one scenario (NOT normalized).

        S1 = Δ1_I + Δ1_G + Δ1_Y
        S2 = Δ2_I + Δ2_G + Δ2_Y
        S3 = Δ3_I + Δ3_G + Δ3_Y
        PS_raw = w1*S1 + w2*S2 + w3*S3
    """
    w1, w2, w3 = params.weights
    s1 = (
        params.s1_effect.get(isf_action, 0.0)
        + params.s1_effect.get(gov_policy, 0.0)
        + params.s1_effect.get(yazd_action, 0.0)
    )
    s2 = (
        params.s2_effect.get(isf_action, 0.0)
        + params.s2_effect.get(gov_policy, 0.0)
        + params.s2_effect.get(yazd_action, 0.0)
    )
    s3 = (
        params.s3_effect.get(isf_action, 0.0)
        + params.s3_effect.get(gov_policy, 0.0)
        + params.s3_effect.get(yazd_action, 0.0)
    )
    return float(w1 * s1 + w2 * s2 + w3 * s3)


def build_public_satisfaction_scores_gov_for_scenarios(
        params: PublicSatisfactionGovParams,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Government public satisfaction & social stability.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
        Higher score means higher public satisfaction / social stability (better for Government).
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                ps = public_satisfaction_index_gov_raw(params, G, I, Y)
                keys.append(scen)
                rows.append(ps)

    ps_arr = np.array(rows, dtype=float)
    score_arr = min_max_normalize(ps_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}


# ---------------------------------------------------------------------
# 12) Security risk (Government) - expert-judgment additive *risk* model
# ---------------------------------------------------------------------
@dataclass
class SecurityRiskGovParams:
    """
    Expert-judgment parameters for Government security risk.

    This is a *risk* indicator (higher = worse), modeled additively around a neutral baseline:

        R = R0 + Δ_I(s_I) + Δ_G(s_G) + Δ_Y(s_Y)

    where R0 is set to 0.0.

    Notes
    -----
    - This is a Government-only criterion: the indicator reflects Government's security-risk exposure.
    - The effects table is typically elicited from expert judgment (or event data proxies if available).
    - In the utility aggregation, LOWER risk should map to HIGHER utility. Therefore, the scenario
      scores returned by `build_security_risk_scores_gov_for_scenarios` are direction-reversed
      (min-max on risk, then 100 - normalized risk).
    """
    isf_effect: Dict[IsfAction, float]
    gov_effect: Dict[GovPolicy, float]
    yazd_effect: Dict[YazdAction, float]
    base: float = 0.0


def default_security_risk_gov_params(
        *,
        excel_path: str | None = None,
) -> SecurityRiskGovParams:
    """
    Load default effects for Government security risk.

    If excel_path is provided, expects an Excel sheet with columns:
      - Actor, Strategy, Delta_Security_Risk

    and actors like: Esfahan, Yazd, Government.

    If excel_path is None, uses the same defaults as the provided impact table,
    with Government R_G treated as 0.0 if left blank.

    This keeps the code runnable even if the Excel file is not present.
    """
    # Start with safe defaults
    isf_effect = {"S_I": 0.0, "M_I": 4.0, "D_I": 8.0}
    yazd_effect = {"S_Y": 0.0, "A_Y": 1.0}
    gov_effect = {"Q_G": 2.0, "C_G": -2.0, "R_G": 0.0}

    if excel_path:
        try:
            df = pd.read_excel(excel_path)
            # Defensive: normalize keys
            for _, row in df.iterrows():
                actor = str(row.get("Actor", "")).strip().lower()
                strat = str(row.get("Strategy", "")).strip()
                delta = row.get("Delta_Security_Risk", 0.0)
                delta = 0.0 if (pd.isna(delta) or delta == "") else float(delta)

                if actor == "esfahan":
                    isf_effect[strat] = delta
                elif actor == "yazd":
                    yazd_effect[strat] = delta
                elif actor == "government":
                    gov_effect[strat] = delta
        except Exception:
            # If Excel read fails, fall back to defaults silently (documented in report)
            pass

    # Ensure required keys exist
    for k in I_STRATS:
        isf_effect.setdefault(k, 0.0)
    for k in Y_STRATS:
        yazd_effect.setdefault(k, 0.0)
    for k in ("Q_G", "C_G", "R_G"):
        gov_effect.setdefault(k, 0.0)

    return SecurityRiskGovParams(
        isf_effect=isf_effect,
        gov_effect=gov_effect,
        yazd_effect=yazd_effect,
        base=0.0,
    )


def security_risk_index_gov(
        params: SecurityRiskGovParams,
        gov_policy: GovPolicy,
        isf_action: IsfAction,
        yazd_action: YazdAction,
) -> float:
    """
    Compute Government security risk index (NOT normalized) for one scenario.

    R = base + Δ_I(s_I) + Δ_G(s_G) + Δ_Y(s_Y)

    Higher values = higher security risk (worse for Government).
    """
    return float(
        params.base
        + params.isf_effect.get(isf_action, 0.0)
        + params.gov_effect.get(gov_policy, 0.0)
        + params.yazd_effect.get(yazd_action, 0.0)
    )


def _min_max_normalize_inverse(values: np.ndarray) -> np.ndarray:
    """
    Normalize a *cost/risk* array to [0,100] as a *benefit score*:
        score = 100 * (1 - (v - vmin)/(vmax - vmin))

    If all values are equal, returns 100 for all (no scenario is worse).
    """
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return np.full_like(values, 100.0, dtype=float)
    return 100.0 * (1.0 - (values - vmin) / (vmax - vmin))


def build_security_risk_scores_gov_for_scenarios(
        params: SecurityRiskGovParams,
) -> Dict[str, float]:
    """
    Produce a score dictionary for Government security risk over the game's scenario naming.

    Outputs 18 keys like:
      "S_I_C_G_S_Y", ... "D_I_R_G_A_Y"

    Returns
    -------
    dict : scenario_name -> score in [0, 100]
        Higher score means LOWER risk (better for Government).
    """
    rows = []
    keys = []

    for I in I_STRATS:
        for G in G_STRATS:
            for Y in Y_STRATS:
                scen = f"{I}_{G}_{Y}"
                risk = security_risk_index_gov(params, G, I, Y)
                keys.append(scen)
                rows.append(risk)

    risk_arr = np.array(rows, dtype=float)
    score_arr = _min_max_normalize_inverse(risk_arr)

    return {keys[i]: float(score_arr[i]) for i in range(len(keys))}


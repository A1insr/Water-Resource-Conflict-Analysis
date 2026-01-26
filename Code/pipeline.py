# pipeline.py
from ahp_utils import ahp_weights, compute_utility, normalize_to_minus5_plus5

# -------------------------------------------------------------------
# SCENARIOS (3 × 3 × 2 = 18)
# Naming: I (Isfahan), G (Government), Y (Yazd)
# S_I / M_I / D_I   = Silence / Protest / Destructive action (Isfahan)
# C_G / Q_G / R_G   = Compensation / Status Quo / Reallocation (Gov)
# S_Y / A_Y         = Silence / Protest (Yazd)
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
# CRITERIA
# -------------------------------------------------------------------
criteria_I = [
    "agricultural_income",
    "social_satisfaction",
    "environment"
]

criteria_Y = [
    "drinking_water_security",
    "industrial_stability",
    "justice_feeling"
]

criteria_G = [
    "budget_cost",        # higher = lower cost / better for gov
    "economic_growth",
    "public_satisfaction",
    "security_risk"       # higher = safer / lower actual risk
]

# -------------------------------------------------------------------
# AHP PAIRWISE MATRICES (EXPERT-BASED, EXAMPLE)
# -------------------------------------------------------------------
pairwise_I = [
    [1,   3,   6],
    [1/3, 1,   2],
    [1/6, 1/2, 1]
]

pairwise_Y = [
    [1,   5,   7],
    [1/5, 1,   2],
    [1/7, 1/2, 1]
]

pairwise_G = [
    [1,   3,   5,   1/4],
    [1/3, 1,   2,   1/6],
    [1/5, 1/2, 1,   1/8],
    [4,   6,   8,   1]
]

# -------------------------------------------------------------------
# FULL SCORES FOR ALL 18 SCENARIOS – ISFAHAN
# 0 = very bad for Isfahan, 100 = very good
# -------------------------------------------------------------------
scores_I = {
    # 1) S_I_C_G_S_Y : Isfahan silent, Gov compensates, Yazd silent
    "S_I_C_G_S_Y": {
        "agricultural_income": 55,
        "social_satisfaction": 60,
        "environment": 35
    },

    # 2) S_I_C_G_A_Y : Isfahan silent, Gov compensates, Yazd protests
    "S_I_C_G_A_Y": {
        "agricultural_income": 55,
        "social_satisfaction": 55,
        "environment": 35
    },

    # 3) S_I_Q_G_S_Y : pure status quo, Isfahan silent
    "S_I_Q_G_S_Y": {
        "agricultural_income": 40,
        "social_satisfaction": 35,
        "environment": 30
    },

    # 4) S_I_Q_G_A_Y : status quo, Yazd protests
    "S_I_Q_G_A_Y": {
        "agricultural_income": 40,
        "social_satisfaction": 30,
        "environment": 30
    },

    # 5) S_I_R_G_S_Y : reallocation in favour of Isfahan, no protests
    "S_I_R_G_S_Y": {
        "agricultural_income": 70,
        "social_satisfaction": 65,
        "environment": 55
    },

    # 6) S_I_R_G_A_Y : reallocation, Yazd protests
    "S_I_R_G_A_Y": {
        "agricultural_income": 70,
        "social_satisfaction": 60,
        "environment": 55
    },

    # 7) M_I_C_G_S_Y : Isfahan protests, Gov compensates, Yazd silent
    "M_I_C_G_S_Y": {
        "agricultural_income": 60,
        "social_satisfaction": 80,
        "environment": 35
    },

    # 8) M_I_C_G_A_Y : same but Yazd protests
    "M_I_C_G_A_Y": {
        "agricultural_income": 60,
        "social_satisfaction": 75,
        "environment": 35
    },

    # 9) M_I_Q_G_S_Y : Isfahan protests, Gov keeps status quo
    "M_I_Q_G_S_Y": {
        "agricultural_income": 35,
        "social_satisfaction": 25,
        "environment": 30
    },

    # 10) M_I_Q_G_A_Y : both Isfahan protests and Yazd protests
    "M_I_Q_G_A_Y": {
        "agricultural_income": 35,
        "social_satisfaction": 20,
        "environment": 30
    },

    # 11) M_I_R_G_S_Y : Isfahan protests, Gov reallocates, Yazd silent
    "M_I_R_G_S_Y": {
        "agricultural_income": 85,
        "social_satisfaction": 90,
        "environment": 70
    },

    # 12) M_I_R_G_A_Y : Isfahan protests, Gov reallocates, Yazd protests
    "M_I_R_G_A_Y": {
        "agricultural_income": 83,
        "social_satisfaction": 88,
        "environment": 68
    },

    # 13) D_I_C_G_S_Y : sabotage by Isfahan, Gov compensates, Yazd silent
    "D_I_C_G_S_Y": {
        "agricultural_income": 45,
        "social_satisfaction": 50,
        "environment": 25
    },

    # 14) D_I_C_G_A_Y : sabotage + Yazd protests
    "D_I_C_G_A_Y": {
        "agricultural_income": 40,
        "social_satisfaction": 45,
        "environment": 25
    },

    # 15) D_I_Q_G_S_Y : sabotage, Gov keeps status quo, Yazd silent
    "D_I_Q_G_S_Y": {
        "agricultural_income": 35,
        "social_satisfaction": 40,
        "environment": 20
    },

    # 16) D_I_Q_G_A_Y : worst: sabotage + status quo + protests in Yazd
    "D_I_Q_G_A_Y": {
        "agricultural_income": 30,
        "social_satisfaction": 35,
        "environment": 20
    },

    # 17) D_I_R_G_S_Y : sabotage, Gov reallocates, Yazd silent
    "D_I_R_G_S_Y": {
        "agricultural_income": 75,
        "social_satisfaction": 60,
        "environment": 40
    },

    # 18) D_I_R_G_A_Y : sabotage, Gov reallocates, Yazd protests
    "D_I_R_G_A_Y": {
        "agricultural_income": 72,
        "social_satisfaction": 55,
        "environment": 40
    },
}

# -------------------------------------------------------------------
# FULL SCORES FOR ALL 18 SCENARIOS – YAZD
# -------------------------------------------------------------------
scores_Y = {
    "S_I_C_G_S_Y": {
        "drinking_water_security": 85,
        "industrial_stability": 80,
        "justice_feeling": 70
    },
    "S_I_C_G_A_Y": {
        "drinking_water_security": 80,
        "industrial_stability": 75,
        "justice_feeling": 55
    },
    "S_I_Q_G_S_Y": {
        "drinking_water_security": 85,
        "industrial_stability": 80,
        "justice_feeling": 65
    },
    "S_I_Q_G_A_Y": {
        "drinking_water_security": 80,
        "industrial_stability": 75,
        "justice_feeling": 50
    },
    "S_I_R_G_S_Y": {
        "drinking_water_security": 75,
        "industrial_stability": 70,
        "justice_feeling": 55
    },
    "S_I_R_G_A_Y": {
        "drinking_water_security": 60,
        "industrial_stability": 60,
        "justice_feeling": 40
    },
    "M_I_C_G_S_Y": {
        "drinking_water_security": 85,
        "industrial_stability": 80,
        "justice_feeling": 65
    },
    "M_I_C_G_A_Y": {
        "drinking_water_security": 80,
        "industrial_stability": 75,
        "justice_feeling": 50
    },
    "M_I_Q_G_S_Y": {
        "drinking_water_security": 85,
        "industrial_stability": 80,
        "justice_feeling": 60
    },
    "M_I_Q_G_A_Y": {
        "drinking_water_security": 80,
        "industrial_stability": 75,
        "justice_feeling": 55
    },
    "M_I_R_G_S_Y": {
        "drinking_water_security": 60,
        "industrial_stability": 65,
        "justice_feeling": 55
    },
    "M_I_R_G_A_Y": {
        "drinking_water_security": 55,
        "industrial_stability": 60,
        "justice_feeling": 40
    },
    "D_I_C_G_S_Y": {
        "drinking_water_security": 60,
        "industrial_stability": 55,
        "justice_feeling": 50
    },
    "D_I_C_G_A_Y": {
        "drinking_water_security": 45,
        "industrial_stability": 45,
        "justice_feeling": 40
    },
    "D_I_Q_G_S_Y": {
        "drinking_water_security": 65,
        "industrial_stability": 60,
        "justice_feeling": 50
    },
    "D_I_Q_G_A_Y": {
        "drinking_water_security": 50,
        "industrial_stability": 50,
        "justice_feeling": 40
    },
    "D_I_R_G_S_Y": {
        "drinking_water_security": 70,
        "industrial_stability": 65,
        "justice_feeling": 45
    },
    "D_I_R_G_A_Y": {
        "drinking_water_security": 55,
        "industrial_stability": 55,
        "justice_feeling": 35
    },
}

# -------------------------------------------------------------------
# FULL SCORES FOR ALL 18 SCENARIOS – GOVERNMENT
# -------------------------------------------------------------------
scores_G = {
    "S_I_C_G_S_Y": {
        "budget_cost": 60,
        "economic_growth": 70,
        "public_satisfaction": 65,
        "security_risk": 70
    },
    "S_I_C_G_A_Y": {
        "budget_cost": 55,
        "economic_growth": 65,
        "public_satisfaction": 55,
        "security_risk": 55
    },
    "S_I_Q_G_S_Y": {
        "budget_cost": 80,
        "economic_growth": 65,
        "public_satisfaction": 40,
        "security_risk": 50
    },
    "S_I_Q_G_A_Y": {
        "budget_cost": 75,
        "economic_growth": 60,
        "public_satisfaction": 35,
        "security_risk": 40
    },
    "S_I_R_G_S_Y": {
        "budget_cost": 65,
        "economic_growth": 72,
        "public_satisfaction": 60,
        "security_risk": 65
    },
    "S_I_R_G_A_Y": {
        "budget_cost": 60,
        "economic_growth": 65,
        "public_satisfaction": 45,
        "security_risk": 45
    },
    "M_I_C_G_S_Y": {
        "budget_cost": 55,
        "economic_growth": 68,
        "public_satisfaction": 60,
        "security_risk": 60
    },
    "M_I_C_G_A_Y": {
        "budget_cost": 50,
        "economic_growth": 60,
        "public_satisfaction": 45,
        "security_risk": 45
    },
    "M_I_Q_G_S_Y": {
        "budget_cost": 80,
        "economic_growth": 60,
        "public_satisfaction": 30,
        "security_risk": 35
    },
    "M_I_Q_G_A_Y": {
        "budget_cost": 75,
        "economic_growth": 55,
        "public_satisfaction": 25,
        "security_risk": 25
    },
    "M_I_R_G_S_Y": {
        "budget_cost": 65,
        "economic_growth": 78,
        "public_satisfaction": 82,
        "security_risk": 58
    },
    "M_I_R_G_A_Y": {
        "budget_cost": 60,
        "economic_growth": 72,
        "public_satisfaction": 65,
        "security_risk": 45
    },
    "D_I_C_G_S_Y": {
        "budget_cost": 50,
        "economic_growth": 55,
        "public_satisfaction": 35,
        "security_risk": 30
    },
    "D_I_C_G_A_Y": {
        "budget_cost": 45,
        "economic_growth": 50,
        "public_satisfaction": 30,
        "security_risk": 20
    },
    "D_I_Q_G_S_Y": {
        "budget_cost": 70,
        "economic_growth": 55,
        "public_satisfaction": 30,
        "security_risk": 35
    },
    "D_I_Q_G_A_Y": {
        "budget_cost": 65,
        "economic_growth": 50,
        "public_satisfaction": 20,
        "security_risk": 15
    },
    "D_I_R_G_S_Y": {
        "budget_cost": 55,
        "economic_growth": 65,
        "public_satisfaction": 45,
        "security_risk": 35
    },
    "D_I_R_G_A_Y": {
        "budget_cost": 50,
        "economic_growth": 60,
        "public_satisfaction": 35,
        "security_risk": 25
    },
}

# -------------------------------------------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------------------------------------------
def run_ahp_pipeline(players_data, normalize_utilities=False):
    """
    Compute AHP weights and utilities for all players and scenarios.
    """
    results = {}

    for player in players_data:
        name = player["name"]
        criteria = player["criteria"]
        pairwise = player["pairwise"]
        scores_dict = player["scores"]

        weight_vector, norm_matrix = ahp_weights(pairwise)
        weight_dict = {criteria[i]: weight_vector[i] for i in range(len(criteria))}

        utilities_raw = {}
        for scenario_name, score_dict in scores_dict.items():
            utilities_raw[scenario_name] = compute_utility(score_dict, weight_dict)

        utilities_scaled = None
        if normalize_utilities:
            raw_vals = list(utilities_raw.values())
            scaled_vals = normalize_to_minus5_plus5(raw_vals)
            utilities_scaled = {
                scenario: scaled_vals[i]
                for i, scenario in enumerate(utilities_raw.keys())
            }

        results[name] = {
            "weights": weight_dict,
            "norm_matrix": norm_matrix,
            "utilities_raw": utilities_raw,
            "utilities_scaled": utilities_scaled
        }

    return results

# -------------------------------------------------------------------
# EXAMPLE RUN (optional)
# -------------------------------------------------------------------
if __name__ == "__main__":
    players_input = [
        {
            "name": "Isfahan",
            "criteria": criteria_I,
            "pairwise": pairwise_I,
            "scores": scores_I
        },
        {
            "name": "Yazd",
            "criteria": criteria_Y,
            "pairwise": pairwise_Y,
            "scores": scores_Y
        },
        {
            "name": "Government",
            "criteria": criteria_G,
            "pairwise": pairwise_G,
            "scores": scores_G
        }
    ]

    results = run_ahp_pipeline(players_input, normalize_utilities=True)

    for player, res in results.items():
        print("\n=== Results for", player, "===")
        print("Weights:", res["weights"])
        print("Raw utilities:", res["utilities_raw"])
        print("Scaled utilities:", res["utilities_scaled"])

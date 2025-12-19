import pandas as pd

from pipeline import (
    run_ahp_pipeline,
    criteria_I, criteria_Y, criteria_G,
    pairwise_I, pairwise_Y, pairwise_G,
    scores_I, scores_Y, scores_G
)

def compute_results(normalize_utilities: bool = False):
    """
    Run the AHP + utility pipeline and return results dict.
    """
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

    results = run_ahp_pipeline(players_input, normalize_utilities=normalize_utilities)
    return results


def build_payoff_tables(results, use_scaled=False):
    """
    Build 3x3 payoff tables for:
      - Yazd = S_Y
      - Yazd = A_Y

    Each cell contains (U_Isfahan, U_Government) for that scenario.
    """

    I_STRATS = ["S_I", "M_I", "D_I"]
    G_STRATS = ["C_G", "Q_G", "R_G"]

    util_key = "utilities_scaled" if use_scaled else "utilities_raw"

    # ---------- Table for Yazd = S_Y ----------
    rows_S = []
    for I in I_STRATS:
        row_dict = {}
        for G in G_STRATS:
            scenario = f"{I}_{G}_S_Y"
            U_I = results["Isfahan"][util_key][scenario]
            U_G = results["Government"][util_key][scenario]
            row_dict[G] = f"({U_I:.2f}, {U_G:.2f})"
        rows_S.append(row_dict)

    df_Y_S = pd.DataFrame(rows_S, index=I_STRATS)
    df_Y_S.index.name = "Isfahan"
    df_Y_S.columns.name = "Government"

    # ---------- Table for Yazd = A_Y ----------
    rows_A = []
    for I in I_STRATS:
        row_dict = {}
        for G in G_STRATS:
            scenario = f"{I}_{G}_A_Y"
            U_I = results["Isfahan"][util_key][scenario]
            U_G = results["Government"][util_key][scenario]
            row_dict[G] = f"({U_I:.2f}, {U_G:.2f})"
        rows_A.append(row_dict)

    df_Y_A = pd.DataFrame(rows_A, index=I_STRATS)
    df_Y_A.index.name = "Isfahan"
    df_Y_A.columns.name = "Government"

    return df_Y_S, df_Y_A


if __name__ == "__main__":
    # Set to True if you prefer normalized payoffs in [-5, +5]
    use_scaled_utilities = False

    results = compute_results(normalize_utilities=use_scaled_utilities)
    df_Y_S, df_Y_A = build_payoff_tables(results, use_scaled=use_scaled_utilities)

    print("\n=== Payoff Matrix (Yazd = S_Y) ===\n")
    print(df_Y_S)

    print("\n=== Payoff Matrix (Yazd = A_Y) ===\n")
    print(df_Y_A)

    # If you want to export to CSV for thesis/Excel:
    # df_Y_S.to_csv("payoff_Yazd_Silent.csv", encoding="utf-8", index=True)
    # df_Y_A.to_csv("payoff_Yazd_Protest.csv", encoding="utf-8", index=True)

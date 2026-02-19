# debug_steps.py
# Run: python debug_steps.py

import inspect
import payoff_table as pt
import pipeline as pl


# ------------------------ helpers ------------------------
def _round_cell(cell, decimals=2):
    if isinstance(cell, tuple):
        return tuple(round(float(x), decimals) for x in cell)
    if isinstance(cell, dict):
        return {k: round(float(v), decimals) for k, v in cell.items()}
    if isinstance(cell, (int, float)):
        return round(float(cell), decimals)
    return cell


def print_payoff_tables(title, tables, decimals=2):
    print("\n" + "=" * 72)
    print(title)
    for y_action, df in tables.payoff_by_y.items():
        print("-" * 72)
        print(f"Conditioned on Yazd action: {y_action}")
        df2 = df.copy()
        for r in df2.index:
            for c in df2.columns:
                df2.loc[r, c] = _round_cell(df2.loc[r, c], decimals)
        print(df2.to_string())


def make_utilities_dict(results, use_scaled=False):
    utilities = {}
    for player_name, res in results.items():
        key = "utilities_scaled" if use_scaled else "utilities_raw"
        utilities[player_name] = res[key]
    return utilities


def range_report(scores_dict, criteria_list, scenarios):
    out = []
    for c in criteria_list:
        vals = [scores_dict[s][c] for s in scenarios if s in scores_dict]
        if not vals:
            out.append((c, None, None, None))
            continue
        mn, mx = min(vals), max(vals)
        out.append((c, mn, mx, mx - mn))
    return out


def extract_scores_from_build_all_scores(ret):
    """
    Accepts different return formats and extracts:
    scores_I, scores_Y, scores_G (each: dict[scenario] -> dict[criterion]->score)
    """
    # Case A: tuple/list of three dicts
    if isinstance(ret, (tuple, list)) and len(ret) == 3:
        return ret[0], ret[1], ret[2]

    # Case B: dict with obvious keys
    if isinstance(ret, dict):
        # common key styles
        keys = set(ret.keys())
        # Try direct names
        for kI, kY, kG in [
            ("scores_I", "scores_Y", "scores_G"),
            ("Isfahan", "Yazd", "Government"),
            ("isfahan", "yazd", "government"),
            ("I", "Y", "G"),
        ]:
            if kI in keys and kY in keys and kG in keys:
                return ret[kI], ret[kY], ret[kG]

    raise TypeError(
        "Could not parse output of build_all_scores(cfg). "
        "Please paste what type/keys it returns."
    )




def main():
    # Must exist per your traceback
    SCENARIOS = pl.SCENARIOS
    criteria_I = pl.criteria_I
    criteria_Y = pl.criteria_Y
    criteria_G = pl.criteria_G
    pairwise_I = pl.pairwise_I
    pairwise_Y = pl.pairwise_Y
    pairwise_G = pl.pairwise_G

    # ---------------- STEP 0: Build scores (this was the missing part) ----------------
    scores_I, scores_Y, scores_G = build_scores()

    players_input = [
        {"name": "Isfahan", "criteria": criteria_I, "pairwise": pairwise_I, "scores": scores_I},
        {"name": "Yazd", "criteria": criteria_Y, "pairwise": pairwise_Y, "scores": scores_Y},
        {"name": "Government", "criteria": criteria_G, "pairwise": pairwise_G, "scores": scores_G},
    ]

    # ---------------- STEP 1: RAW utilities (no normalization) + payoff ----------------
    results_raw = pl.run_ahp_pipeline(players_input, normalize_utilities=False)
    utilities_raw = make_utilities_dict(results_raw, use_scaled=False)

    sig = inspect.signature(pt.build_payoff_tables)
    if "decimals" in sig.parameters:
        tables_raw = pt.build_payoff_tables(utilities_raw, decimals=2)
    else:
        tables_raw = pt.build_payoff_tables(utilities_raw)

    print_payoff_tables("STEP 1) PAYOFF TABLES (RAW utilities, no normalization)", tables_raw, decimals=2)

    # ---------------- STEP 2: Print AHP weights ----------------
    print("\n" + "=" * 72)
    print("STEP 2) AHP WEIGHTS")
    for player_name, res in results_raw.items():
        print(f"\n--- {player_name} ---")
        for k, v in res["weights"].items():
            print(f"{k:28s} : {v:.6f}")

    # ---------------- STEP 3: Range report ----------------
    print("\n" + "=" * 72)
    print("STEP 3) RANGE REPORT (min / max / range) across scenarios")

    print("\n--- Isfahan criteria ranges ---")
    for c, mn, mx, rg in range_report(scores_I, criteria_I, SCENARIOS):
        if mn is None:
            print(f"{c:28s}  (no values found)")
        else:
            print(f"{c:28s}  min={mn:7.2f}  max={mx:7.2f}  range={rg:7.2f}")

    print("\n--- Yazd criteria ranges ---")
    for c, mn, mx, rg in range_report(scores_Y, criteria_Y, SCENARIOS):
        if mn is None:
            print(f"{c:28s}  (no values found)")
        else:
            print(f"{c:28s}  min={mn:7.2f}  max={mx:7.2f}  range={rg:7.2f}")

    print("\n--- Government criteria ranges ---")
    for c, mn, mx, rg in range_report(scores_G, criteria_G, SCENARIOS):
        if mn is None:
            print(f"{c:28s}  (no values found)")
        else:
            print(f"{c:28s}  min={mn:7.2f}  max={mx:7.2f}  range={rg:7.2f}")


if __name__ == "__main__":
    main()




"""
Payoff table utilities for the Isfahan–Yazd–Government water conflict game.

Creates two 3×3 payoff tables (I×G) per Yazd strategy (S_Y or A_Y) so that the 3-player
game can be inspected as two normal-form games conditioned on Yazd's action.

Each cell contains a tuple of utilities (U_I, U_G, U_Y) or a dict, depending on output mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple, Any

import pandas as pd

IsfAction = Literal["S_I", "M_I", "D_I"]
GovPolicy = Literal["C_G", "Q_G", "R_G"]
YazdAction = Literal["S_Y", "A_Y"]

I_STRATS: Tuple[IsfAction, ...] = ("S_I", "M_I", "D_I")
G_STRATS: Tuple[GovPolicy, ...] = ("C_G", "Q_G", "R_G")
Y_STRATS: Tuple[YazdAction, ...] = ("S_Y", "A_Y")


@dataclass(frozen=True)
class PayoffTables:
    """
    payoff_by_y: mapping Yazd strategy -> 3x3 DataFrame indexed by Isfahan action and columns Gov policy.
    Each cell is typically a tuple (U_Isfahan, U_Government, U_Yazd).
    """
    payoff_by_y: Dict[YazdAction, pd.DataFrame]


def _scenario(I: IsfAction, G: GovPolicy, Y: YazdAction) -> str:
    return f"{I}_{G}_{Y}"


def _round_cell(cell: Any, decimals: int) -> Any:
    if isinstance(cell, tuple):
        return tuple(round(float(x), decimals) for x in cell)
    if isinstance(cell, dict):
        return {k: round(float(v), decimals) for k, v in cell.items()}
    if isinstance(cell, (float, int)):
        return round(float(cell), decimals)
    return cell


def build_payoff_tables(
    utilities: Dict[str, Dict[str, float]],
    *,
    order_players: Tuple[str, str, str] = ("Isfahan", "Government", "Yazd"),
    cell_mode: Literal["tuple", "dict"] = "tuple",
    decimals: int | None = 2,
) -> PayoffTables:
    """
    Build two 3x3 payoff tables from per-scenario utilities.

    Parameters
    ----------
    utilities:
        Dict[player_name][scenario] -> utility (already aggregated across criteria).
        Example keys for players: "Isfahan", "Yazd", "Government".
    order_players:
        The order of utilities to place in each cell.
        Default: (Isfahan, Government, Yazd) for easier reading in a 3-player table.
    cell_mode:
        - "tuple": each cell is a tuple ordered as order_players
        - "dict":  each cell is a dict {player: utility}
    decimals:
        If not None, round utilities inside each cell to this many decimals **at construction time**.
        (This guarantees printing/inspection shows rounded values, regardless of how you print the DataFrame.)

    Returns
    -------
    PayoffTables with two DataFrames (for S_Y and A_Y).
    """
    missing = [p for p in order_players if p not in utilities]
    if missing:
        raise KeyError(f"utilities is missing players: {missing}. Present: {list(utilities.keys())}")

    payoff_by_y: Dict[YazdAction, pd.DataFrame] = {}

    for Y in Y_STRATS:
        rows = []
        for I in I_STRATS:
            row = []
            for G in G_STRATS:
                scen = _scenario(I, G, Y)
                if cell_mode == "dict":
                    cell: Any = {p: float(utilities[p][scen]) for p in order_players}
                else:
                    cell = tuple(float(utilities[p][scen]) for p in order_players)

                if decimals is not None:
                    cell = _round_cell(cell, decimals)

                row.append(cell)
            rows.append(row)

        df = pd.DataFrame(rows, index=list(I_STRATS), columns=list(G_STRATS))
        df.index.name = "Isfahan"
        df.columns.name = "Government"
        payoff_by_y[Y] = df

    return PayoffTables(payoff_by_y=payoff_by_y)


def format_payoff_tables_for_print(
    tables: PayoffTables,
    *,
    header: str = "Payoff tables (each cell: (U_Isfahan, U_Government, U_Yazd))",
) -> str:
    """Readable string representation for console logs / thesis appendix drafts."""
    lines = [header]
    for y, df in tables.payoff_by_y.items():
        lines.append("\n" + "-" * 72)
        lines.append(f"Conditioned on Yazd action: {y}")
        lines.append(df.to_string())
    return "\n".join(lines)

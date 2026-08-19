"""Score a projection by the squad it actually tells you to buy.

Spearman rho over the whole player pool is a poor objective for FPL. It rewards
correctly ordering the 500th and 550th most valuable players, which no manager
ever has to decide. Two variants can differ by 0.02 rho while recommending
squads that differ by 200 real points.

The decision-relevant question is narrower: if I hand the projection to the
optimiser and buy what it says, how many points do I actually get? That is what
this measures, using real prices at the target season's GW1 deadline and real
final points as the answer key.

A reference squad picked by last season's points gives the beat-the-baseline bar.
A hindsight-optimal squad gives the ceiling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 100.0
TEAM_LIMIT = 3
BENCH_WEIGHT = 0.15


def pick_squad(d: pd.DataFrame, value: np.ndarray,
               budget: float = BUDGET) -> np.ndarray:
    """Return a boolean mask of the optimal 15 under FPL squad rules."""
    d = d.reset_index(drop=True)
    n = len(d)
    price = d["price"].to_numpy(float)
    ep = np.nan_to_num(value, nan=0.0)

    c = -np.concatenate([ep * BENCH_WEIGHT, ep * (1.0 - BENCH_WEIGHT)])
    rows, lbs, ubs = [], [], []

    def add(row, lo, hi):
        rows.append(row)
        lbs.append(lo)
        ubs.append(hi)

    for pos, k in SQUAD.items():
        r = np.zeros(2 * n)
        r[:n] = (d["pos"] == pos).to_numpy(float)
        add(r, k, k)
    r = np.zeros(2 * n)
    r[:n] = price
    add(r, 0, budget)
    for team in d["team_s"].dropna().unique():
        r = np.zeros(2 * n)
        r[:n] = (d["team_s"] == team).to_numpy(float)
        add(r, 0, TEAM_LIMIT)
    r = np.zeros(2 * n)
    r[n:] = 1.0
    add(r, 11, 11)
    for pos in SQUAD:
        r = np.zeros(2 * n)
        r[n:] = (d["pos"] == pos).to_numpy(float)
        add(r, XI_MIN[pos], XI_MAX[pos])
    for i in range(n):
        r = np.zeros(2 * n)
        r[i] = 1.0
        r[n + i] = -1.0
        add(r, 0, np.inf)

    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lbs, ubs),
               integrality=np.ones(2 * n), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"solver failed: {res.message}")
    return np.round(res.x[:n]).astype(bool)


def squad_score(d: pd.DataFrame, value: np.ndarray,
                budget: float = BUDGET) -> dict:
    """Buy what `value` recommends, then score it on real outcomes."""
    ok = d["pos"].notna() & d["price"].notna() & (d["price"] > 0)
    sub = d[ok].reset_index(drop=True)
    val = np.asarray(value)[ok.to_numpy()]
    mask = pick_squad(sub, val, budget)
    squad = sub[mask]
    return {
        "squad_actual": float(squad["actual"].sum()),
        "squad_cost": float(squad["price"].sum()),
        "squad_names": ", ".join(squad.nlargest(6, "actual")["web_name"].astype(str)),
    }


def reference_scores(d: pd.DataFrame) -> dict:
    """The baseline bar and the hindsight ceiling."""
    out = {}
    bl = squad_score(d, d["baseline"].to_numpy(float))
    out["baseline_squad_actual"] = bl["squad_actual"]
    ceil = squad_score(d, d["actual"].to_numpy(float))
    out["ceiling_squad_actual"] = ceil["squad_actual"]
    return out

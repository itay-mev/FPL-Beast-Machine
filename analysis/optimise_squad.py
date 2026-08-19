"""Pick the optimal 15-man FPL squad under the 2026/27 rules by integer program.

Two decisions are made jointly: which 15 to own, and which 11 of them start.
Doing both at once matters, because a squad that maximises total points often
buries value on the bench where it cannot score. Bench players are still worth
something (injury cover, rotation, Bench Boost), so they carry a reduced weight
rather than zero.

Constraints implemented from docs/FPL_2026_27_RULES.md:
  budget 100.0, squad 2/5/5/3, max 3 per club,
  XI of 11 with 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

BUDGET = 100.0
SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
TEAM_LIMIT = 3
BENCH_WEIGHT = 0.15


def solve(df: pd.DataFrame, value_col: str, budget: float = BUDGET,
          locked: list | None = None, banned: list | None = None) -> pd.DataFrame:
    d = df.reset_index(drop=True)
    n = len(d)
    ep = d[value_col].to_numpy(float)
    price = d["price"].to_numpy(float)

    # Decision vector is [x_0..x_n-1 squad, y_0..y_n-1 starting XI].
    c = -np.concatenate([ep * BENCH_WEIGHT, ep * (1.0 - BENCH_WEIGHT)])

    rows, lbs, ubs = [], [], []

    def add(row, lo, hi):
        rows.append(row)
        lbs.append(lo)
        ubs.append(hi)

    # Squad composition and budget.
    for pos, k in SQUAD.items():
        r = np.zeros(2 * n)
        r[:n] = (d["pos"] == pos).to_numpy(float)
        add(r, k, k)
    r = np.zeros(2 * n)
    r[:n] = price
    add(r, 0, budget)

    # Max three players from any one club.
    for team in d["team_s"].dropna().unique():
        r = np.zeros(2 * n)
        r[:n] = (d["team_s"] == team).to_numpy(float)
        add(r, 0, TEAM_LIMIT)

    # Starting XI size and shape.
    r = np.zeros(2 * n)
    r[n:] = 1.0
    add(r, 11, 11)
    for pos in SQUAD:
        r = np.zeros(2 * n)
        r[n:] = (d["pos"] == pos).to_numpy(float)
        add(r, XI_MIN[pos], XI_MAX[pos])

    # A player can only start if they are owned.
    for i in range(n):
        r = np.zeros(2 * n)
        r[i] = 1.0
        r[n + i] = -1.0
        add(r, 0, np.inf)

    for name in (locked or []):
        idx = d.index[d["web_name"] == name]
        if len(idx):
            r = np.zeros(2 * n)
            r[int(idx[0])] = 1.0
            add(r, 1, 1)
    for name in (banned or []):
        idx = d.index[d["web_name"] == name]
        if len(idx):
            r = np.zeros(2 * n)
            r[int(idx[0])] = 1.0
            add(r, 0, 0)

    res = milp(
        c=c,
        constraints=LinearConstraint(np.array(rows), lbs, ubs),
        integrality=np.ones(2 * n),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError(f"solver failed: {res.message}")

    x = np.round(res.x[:n]).astype(bool)
    y = np.round(res.x[n:]).astype(bool)
    out = d[x].copy()
    out["starting"] = y[x]
    return out.sort_values(["starting", "pos", value_col], ascending=[False, True, False])


def report(sq: pd.DataFrame, value_col: str, label: str) -> None:
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    sq = sq.assign(_o=sq["pos"].map(order)).sort_values(
        ["starting", "_o", value_col], ascending=[False, True, False])
    xi = sq[sq["starting"]]
    bench = sq[~sq["starting"]]
    form = "-".join(str((xi["pos"] == p).sum()) for p in ["DEF", "MID", "FWD"])
    print("=" * 92)
    print(f"{label}   cost {sq['price'].sum():.1f}m   formation {form}   "
          f"XI {value_col} {xi[value_col].sum():.1f}")
    print("=" * 92)
    cols = ["web_name", "team_s", "pos", "price", "own", "ep1", value_col]
    show = [c for c in cols if c in sq.columns]
    print("STARTING XI")
    print(xi[show].to_string(index=False))
    print("BENCH")
    print(bench[show].to_string(index=False))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="ep_gw1_6",
                    help="column to maximise (ep_gw1_6 or ep_season)")
    ap.add_argument("--max-own", type=float, default=None,
                    help="cap ownership to force differentials")
    args = ap.parse_args()

    d = pd.read_csv("analysis/out_projections.csv")
    # Only consider players who are fit and have a real chance of playing.
    d = d[(d["avail"] > 0.5) & d["pos"].notna() & (d["exp_mins"] > 8)].copy()

    base = solve(d, args.value)
    report(base, args.value, f"OPTIMAL SQUAD (maximising {args.value})")

    if args.max_own is not None:
        alt = solve(d[d["own"] <= args.max_own], args.value)
        report(alt, args.value, f"DIFFERENTIAL SQUAD (ownership <= {args.max_own}%)")


if __name__ == "__main__":
    main()

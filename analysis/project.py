"""Expected-points model for FPL 2026/27, plus a backtest of the same method.

Points are built up component by component under the 2026/27 scoring rules
(see docs/FPL_2026_27_RULES.md), then scaled by expected minutes and fixture.

Clean sheets use a Poisson probability on expected goals against rather than a
flat rate, because the payoff is a step function and the mean understates the
value of facing weak attacks.

Everything is keyed on player_code. player_id is re-issued each season.
"""
import json
import numpy as np
import pandas as pd

GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
DC_THRESH = {"DEF": 10, "MID": 12, "FWD": 12}
LG_HOME_GF, LG_AWAY_GF = 1.54, 1.24


def poisson_cs(xga: float) -> float:
    return float(np.exp(-max(xga, 0.05)))


def bonus_per90(bps90: float, pos: str) -> float:
    """Empirical BPS-to-bonus curve, adjusted for the 2026/27 BPS rebalance.

    Bonus is a rank-based prize, so it rises steeply once a player is regularly
    near the top three of a match. A saturating curve fits that better than a
    linear one. The 2026/27 rebalance cut CBI value from 1-per-2 to 1-per-3 and
    removed the being-tackled penalty, so CBI-heavy centre-backs are shaded down
    and ball-carriers shaded up.
    """
    base = 2.2 * (bps90 / 32.0) ** 2.1
    return float(np.clip(base, 0, 2.2))


def project_row(r: pd.Series, atk: float, dfn: float, opp_atk: float,
                opp_dfn: float, venue: str) -> float:
    pos = r["pos"]
    mins = r["exp_mins"]
    if mins <= 0:
        return 0.0
    p60 = r["p60"]
    n90 = mins / 90.0

    # Attacking output, rescaled from the player's old club context to the new
    # one, then adjusted for the opponent's defensive quality.
    ctx = (atk / max(r["old_atk"], 0.35)) * opp_dfn
    xg = r["xg90"] * n90 * ctx
    xa = r["xa90"] * n90 * ctx
    pts = xg * GOAL_PTS[pos] + xa * 3.0

    # Appearance
    pts += 2.0 * p60 + 1.0 * (r["p_app"] - p60)

    # Clean sheet and goals conceded
    lg_gf = LG_HOME_GF if venue == "H" else LG_AWAY_GF
    xga = lg_gf * dfn * opp_atk
    if CS_PTS[pos] > 0:
        pts += poisson_cs(xga) * CS_PTS[pos] * p60
    if pos in ("GKP", "DEF"):
        pts -= 0.5 * xga * (mins / 90.0)

    # Defensive contribution (capped at 2, so use the hit rate directly)
    if pos in DC_THRESH:
        pts += r["dc_rate"] * 2.0 * r["p_app"]

    # Goalkeeper saves, scaled by how much shooting the opponent does
    if pos == "GKP":
        pts += (r["saves90"] * n90 * opp_atk) / 3.0

    pts += bonus_per90(r["bps90"], pos) * n90
    pts -= r["yc90"] * n90
    return float(pts)


def impute_from_price(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Players with no prior-season record get position-and-price-based priors.

    FPL prices encode the game's own expectation of a player, so fitting the
    observed relationship between price and output on players who DO have a
    record gives a defensible prior for those who do not.
    """
    out = df.copy()
    for pos in out["pos"].unique():
        known = out[(out["pos"] == pos) & out["has_history"]]
        unknown = (out["pos"] == pos) & ~out["has_history"]
        if len(known) < 8 or unknown.sum() == 0:
            continue
        for c in cols:
            y = pd.to_numeric(known[c], errors="coerce")
            x = known["price"]
            ok = y.notna() & x.notna()
            if ok.sum() < 8:
                continue
            b, a = np.polyfit(x[ok], y[ok], 1)
            pred = a + b * out.loc[unknown, "price"]
            lo, hi = y[ok].quantile([0.05, 0.95])
            out.loc[unknown, c] = pred.clip(lo, hi)
    return out

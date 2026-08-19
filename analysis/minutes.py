"""Expected-minutes model blending three independent signals.

Last season's minutes alone badly misprice any player whose role changed over
the summer. Isak played 694 Premier League minutes in 2025/26 but started every
Liverpool pre-season friendly, and a history-only model projects him near zero.

Three signals are combined instead:
  1. Prior season   how much they actually played in the Premier League
  2. Pre-season     whether they are starting friendlies for their current club
  3. Price          FPL's own valuation, which encodes expected role

Pre-season is weighted highest when available because it is the only signal that
reflects the current club and current manager. Price acts as the anchor for
players with neither a Premier League record nor friendly coverage.
"""
import numpy as np
import pandas as pd

STARTER_MINS = 78.0   # typical minutes for a player who starts a league match
SUB_MINS = 22.0       # typical minutes for a player who appears off the bench
FRIENDLY_START_MIN = 45


def friendly_signal(season_dir: str = "data/2026-2027") -> pd.DataFrame:
    """Pre-season appearance profile per player, keyed on player_code."""
    pm = pd.read_csv(f"{season_dir}/By Tournament/Friendlies/GW0/playermatchstats.csv")
    pl = pd.read_csv(f"{season_dir}/players.csv")
    pm["minutes_played"] = pd.to_numeric(pm["minutes_played"], errors="coerce").fillna(0)
    pm = pm.merge(pl[["player_id", "player_code"]], on="player_id", how="left")
    played = pm[pm["minutes_played"] > 0]
    g = played.groupby("player_code", as_index=False).agg(
        fr_matches=("minutes_played", "size"),
        fr_mins=("minutes_played", "sum"),
        fr_starts=("minutes_played", lambda s: int((s >= FRIENDLY_START_MIN).sum())),
    )
    g["fr_start_rate"] = g["fr_starts"] / g["fr_matches"]
    return g


def price_implied_start(df: pd.DataFrame) -> pd.Series:
    """Start probability implied by price, fit within each position.

    Fit on players with an established role (20+ prior appearances) so the
    relationship reflects genuine starters rather than injury-hit seasons.
    """
    out = pd.Series(np.nan, index=df.index)
    for pos in df["pos"].dropna().unique():
        sel = df["pos"] == pos
        fit = df[sel & (df["apps"].fillna(0) >= 20)]
        if len(fit) < 8:
            out[sel] = 0.5
            continue
        y = (fit["app60"] / 38.0).clip(0, 1)
        b, a = np.polyfit(fit["price"], y, 1)
        out[sel] = np.clip(a + b * df.loc[sel, "price"], 0.05, 0.95)
    return out


def _apply_gk_depth_chart(d: pd.DataFrame) -> pd.DataFrame:
    """Only one goalkeeper per club plays, so rank them and demote the rest.

    Outfield positions share minutes across a squad, but goalkeeper is close to
    winner-takes-all. Without this, an expensive backup keeper at a strong club
    projects almost as well as the first choice.

    Ranking uses ownership first because it aggregates public knowledge of who
    is first choice, then pre-season starts, then prior-season appearances.
    """
    gk = d["pos"] == "GKP"
    if not gk.any():
        return d
    score = (
        d["own"].fillna(0) * 10.0
        + d["fr_starts"].fillna(0) * 3.0
        + d["apps"].fillna(0) * 0.5
        + d["price"].fillna(4.0)
    )
    d = d.assign(_gk_score=score)
    rank = (
        d[gk].groupby("team_s")["_gk_score"]
        .rank(ascending=False, method="first")
    )
    d.loc[gk, "_gk_rank"] = rank
    # Second choice keeps a small share for injury and cup rotation cover.
    demote = d.loc[gk, "_gk_rank"].map({1.0: 1.0, 2.0: 0.10}).fillna(0.03)
    d.loc[gk, "p_start"] = d.loc[gk, "p_start"] * demote
    return d.drop(columns=["_gk_score"])


def expected_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Attach p_start, p_app and exp_mins to the player frame."""
    d = df.copy()
    d = d.merge(friendly_signal(), on="player_code", how="left")

    p_hist = (d["app60"].fillna(0) / 38.0).clip(0, 1)
    hist_apps = d["apps"].fillna(0)
    fr_matches = d["fr_matches"].fillna(0)
    p_fr = d["fr_start_rate"]
    p_price = price_implied_start(d)

    # Confidence in each signal scales with how much of it we have.
    w_hist = 0.45 * (hist_apps / (hist_apps + 8.0))
    w_fr = 0.55 * (fr_matches / (fr_matches + 2.0))
    w_price = 0.30

    # Absence of evidence is itself evidence. A player with no Premier League
    # record and no pre-season minutes is usually a fringe squad member, so
    # leaning on price alone would promote every expensive backup. This pulls
    # such players down rather than letting price fill the vacuum.
    evidence = hist_apps + 3.0 * fr_matches
    w_none = 0.85 / (1.0 + evidence / 2.5)
    p_none = 0.10

    tot = w_hist + w_fr + w_price + w_none
    d["p_start"] = (
        w_hist * p_hist.fillna(0)
        + w_fr * p_fr.fillna(0)
        + w_price * p_price.fillna(0.4)
        + w_none * p_none
    ) / tot
    d["p_start"] = d["p_start"].clip(0, 0.95)

    d = _apply_gk_depth_chart(d)

    # Bench appearances on top of starts, from the prior-season gap.
    sub_rate = ((d["apps"].fillna(0) - d["app60"].fillna(0)) / 38.0).clip(0, 0.5)
    d["p_app"] = (d["p_start"] + sub_rate.fillna(0.1)).clip(0, 0.98)
    d["p60"] = d["p_start"]
    d["exp_mins"] = d["p_start"] * STARTER_MINS + (d["p_app"] - d["p_start"]) * SUB_MINS
    return d

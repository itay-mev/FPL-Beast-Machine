"""Backtest the projection method: predict 2025/26 using only 2024/25 data.

The 2026/27 projection cannot be validated directly because the season has not
started. Running the identical method one season earlier, where the answer is
already known, is the only honest way to find out whether it is worth anything.

The bar to clear is the naive baseline every FPL manager already uses: last
season's total points. A model that cannot beat that is not adding value.

This backtest runs without the pre-season friendly signal, because 2024/25 has
no friendly data in the repo. It therefore measures the floor of the method's
accuracy rather than its ceiling.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from project import project_row, impute_from_price
from minutes import STARTER_MINS, SUB_MINS, price_implied_start, _apply_gk_depth_chart

POS_SHORT = {"Goalkeeper": "GKP", "Defender": "DEF",
             "Midfielder": "MID", "Forward": "FWD"}
DC_T = {"DEF": 10, "MID": 12, "FWD": 12}
SHRINK_RATE, SHRINK_DC = 8.0, 6.0


def prior_2425() -> pd.DataFrame:
    """2024/25 per-player rates, rebuilt from match data.

    The 2024/25 playerstats table is a narrower 58-column schema with no
    minutes, goals, assists or starts, so those come from playermatchstats.
    """
    pm = pd.read_csv("data/2024-2025/playermatchstats/playermatchstats.csv")
    pl = pd.read_csv("data/2024-2025/players/players.csv")
    ps = pd.read_csv("data/2024-2025/playerstats/playerstats.csv")
    ps = ps[ps["gw"] == 38]

    for c in ["minutes_played", "goals", "assists", "tackles", "clearances",
              "interceptions", "blocks", "recoveries", "saves", "yellow_cards"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce").fillna(0) if c in pm.columns else 0.0

    pm = pm.merge(pl[["player_id", "player_code", "position"]],
                  on="player_id", how="left")
    pm["pos"] = pm["position"].map(POS_SHORT)
    cbit = pm[["tackles", "clearances", "interceptions", "blocks"]].sum(axis=1)
    pm["actions"] = cbit.where(pm["pos"] == "DEF", cbit + pm["recoveries"])
    pm["hit"] = pm["actions"] >= pm["pos"].map(DC_T).fillna(999)

    played = pm[pm["minutes_played"] > 0]
    agg = played.groupby("player_code", as_index=False).agg(
        p_mins=("minutes_played", "sum"),
        apps=("minutes_played", "size"),
        app60=("minutes_played", lambda s: int((s >= 60).sum())),
        p_dc_hits=("hit", "sum"),
        p_apps=("hit", "size"),
        saves_tot=("saves", "sum"),
        yc_tot=("yellow_cards", "sum"),
    )
    agg["p_dc_rate"] = agg["p_dc_hits"] / agg["p_apps"]

    ps = ps.merge(pl[["player_id", "player_code", "team_code"]],
                  left_on="id", right_on="player_id", how="left")
    for c in ["expected_goals", "expected_assists", "bps", "total_points"]:
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)
    fpl = ps.groupby("player_code", as_index=False).agg(
        xg=("expected_goals", "max"), xa=("expected_assists", "max"),
        bps=("bps", "max"), prior_pts=("total_points", "max"),
        p_team_code=("team_code", "first"))

    out = agg.merge(fpl, on="player_code", how="left")
    n90 = (out["p_mins"] / 90.0).clip(lower=0.5)
    out["p_xg90"] = out["xg"] / n90
    out["p_xa90"] = out["xa"] / n90
    out["p_bps90"] = out["bps"] / n90
    out["p_saves90"] = out["saves_tot"] / n90
    out["p_yc90"] = out["yc_tot"] / n90
    return out


def team_strength_2425() -> pd.DataFrame:
    m = pd.read_csv("data/2024-2025/matches/matches.csv").drop_duplicates("match_id")
    names = pd.read_csv("data/2024-2025/teams/teams.csv").set_index("code")["short_name"].to_dict()
    parts = []
    for side, gf, ga, xgf, xga, v in [
        ("home_team", "home_score", "away_score",
         "home_expected_goals_xg", "away_expected_goals_xg", "H"),
        ("away_team", "away_score", "home_score",
         "away_expected_goals_xg", "home_expected_goals_xg", "A")]:
        parts.append(pd.DataFrame({
            "team_s": m[side].map(names), "venue": v,
            "af": 0.7 * pd.to_numeric(m[xgf], errors="coerce").fillna(m[gf]) + 0.3 * m[gf],
            "aa": 0.7 * pd.to_numeric(m[xga], errors="coerce").fillna(m[ga]) + 0.3 * m[ga]}))
    t = pd.concat(parts, ignore_index=True).dropna(subset=["team_s"])
    lg = {v: t[t["venue"] == v][["af", "aa"]].mean() for v in "HA"}
    res = []
    for (team, v), g in t.groupby(["team_s", "venue"]):
        n, la, ld = len(g), lg[v]["af"], lg[v]["aa"]
        res.append({"team_s": team, "venue": v,
                    "attack": (g["af"].sum() + 6 * la) / (n + 6) / la,
                    "defence": (g["aa"].sum() + 6 * ld) / (n + 6) / ld})
    r = pd.DataFrame(res).pivot(index="team_s", columns="venue",
                                values=["attack", "defence"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    return r


def minutes_no_friendly(d: pd.DataFrame) -> pd.DataFrame:
    """Same minutes blend as the live model, minus the unavailable friendly term."""
    d = d.copy()
    p_hist = (d["app60"].fillna(0) / 38.0).clip(0, 1)
    hist_apps = d["apps"].fillna(0)
    p_price = price_implied_start(d)
    w_hist = 0.45 * (hist_apps / (hist_apps + 8.0))
    w_price = 0.30
    w_none = 0.85 / (1.0 + hist_apps / 2.5)
    tot = w_hist + w_price + w_none
    d["p_start"] = ((w_hist * p_hist.fillna(0)
                     + w_price * p_price.fillna(0.4)
                     + w_none * 0.10) / tot).clip(0, 0.95)
    d["fr_starts"] = 0.0
    d = _apply_gk_depth_chart(d)
    sub_rate = ((d["apps"].fillna(0) - d["app60"].fillna(0)) / 38.0).clip(0, 0.5)
    d["p_app"] = (d["p_start"] + sub_rate.fillna(0.1)).clip(0, 0.98)
    d["p60"] = d["p_start"]
    d["exp_mins"] = d["p_start"] * STARTER_MINS + (d["p_app"] - d["p_start"]) * SUB_MINS
    return d


def main() -> None:
    prior = prior_2425()
    ts = team_strength_2425()

    # The 2025/26 pool exactly as it looked at the GW1 deadline.
    pool = pd.read_csv("data/2025-2026/By Gameweek/GW1/playerstats.csv")
    pl26 = pd.read_csv("data/2025-2026/players.csv")
    code2short = pd.read_csv("data/2025-2026/teams.csv").set_index("code")["short_name"].to_dict()
    pool = pool.merge(pl26[["player_id", "player_code", "position", "team_code"]],
                      left_on="id", right_on="player_id", how="left")
    pool["pos"] = pool["position"].map(POS_SHORT)
    pool["team_s"] = pool["team_code"].map(code2short)
    pool["price"] = pd.to_numeric(pool["now_cost"], errors="coerce") / 10.0
    pool["own"] = pd.to_numeric(pool["selected_by_percent"], errors="coerce").fillna(0)

    m = pool.merge(prior, on="player_code", how="left")
    m["has_history"] = m["p_mins"].notna() & (m["p_mins"] > 0)
    old_names = pd.read_csv("data/2024-2025/teams/teams.csv").set_index("code")["short_name"].to_dict()
    m["old_team"] = m["p_team_code"].map(old_names)
    mean_atk = float(((ts["attack_H"] + ts["attack_A"]) / 2).mean())
    m["old_atk"] = m["old_team"].map(
        ((ts["attack_H"] + ts["attack_A"]) / 2).to_dict()).fillna(mean_atk)

    m["n90"] = m["p_mins"].fillna(0) / 90.0
    for src, dst in [("p_xg90", "xg90"), ("p_xa90", "xa90"), ("p_bps90", "bps90"),
                     ("p_yc90", "yc90"), ("p_saves90", "saves90")]:
        m[dst] = pd.to_numeric(m[src], errors="coerce")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        sel = m["pos"] == pos
        for c in ["xg90", "xa90", "bps90", "yc90", "saves90"]:
            mu = m.loc[sel & m["has_history"], c].mean()
            w = m.loc[sel, "n90"] / (m.loc[sel, "n90"] + SHRINK_RATE)
            m.loc[sel, c] = w * m.loc[sel, c].fillna(mu) + (1 - w) * mu

    m["dc_rate"] = pd.to_numeric(m["p_dc_rate"], errors="coerce")
    for pos in DC_T:
        sel = m["pos"] == pos
        mu = m.loc[sel, "dc_rate"].mean()
        a = m.loc[sel, "p_apps"].fillna(0)
        w = a / (a + SHRINK_DC)
        m.loc[sel, "dc_rate"] = w * m.loc[sel, "dc_rate"].fillna(mu) + (1 - w) * mu
    m.loc[m["pos"] == "GKP", "dc_rate"] = 0.0
    m = impute_from_price(m, ["xg90", "xa90", "bps90", "yc90", "saves90", "dc_rate"])
    m["dc_rate"] = m["dc_rate"].clip(0, 0.95)

    m = minutes_no_friendly(m)
    m["avail"] = np.where(m["status"] == "a", 1.0, 0.6)

    # Season totals against a neutral opponent, 19 home and 19 away.
    proj = []
    for _, r in m.iterrows():
        t = r["team_s"]
        if t not in ts.index or pd.isna(r["pos"]):
            proj.append(0.0)
            continue
        tot = sum(19 * project_row(r, ts.loc[t, f"attack_{v}"],
                                   ts.loc[t, f"defence_{v}"], 1.0, 1.0, v)
                  for v in ("H", "A"))
        proj.append(tot * r["avail"])
    m["proj"] = proj

    actual = pd.read_csv("data/2025-2026/By Gameweek/GW38/playerstats.csv")
    actual = actual.merge(pl26[["player_id", "player_code"]],
                          left_on="id", right_on="player_id", how="left")
    act = actual.groupby("player_code")["total_points"].max().rename("actual")
    m = m.merge(act, on="player_code", how="left")
    m["actual"] = pd.to_numeric(m["actual"], errors="coerce").fillna(0)
    m["baseline"] = pd.to_numeric(m["prior_pts"], errors="coerce").fillna(0)
    m.to_csv("analysis/out_backtest.csv", index=False)

    ev = m[m["pos"].notna()].copy()
    best50 = set(ev.nlargest(50, "actual")["player_code"])
    print("=" * 82)
    print("BACKTEST: predict 2025/26 season points using only 2024/25 data")
    print("=" * 82)
    print(f"Players evaluated: {len(ev)}")
    print()
    for label, col in [("MODEL", "proj"), ("BASELINE (last season's points)", "baseline")]:
        rho = spearmanr(ev[col], ev["actual"]).statistic
        top = ev.nlargest(50, col)
        hit = len(set(top["player_code"]) & best50)
        print(f"  {label:32} rho={rho:.3f}  top50 overlap={hit}/50  "
              f"mean actual of picks={top['actual'].mean():.1f}")
    print()
    print("  By position (Spearman rho):")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        s = ev[ev["pos"] == pos]
        rm = spearmanr(s["proj"], s["actual"]).statistic
        rb = spearmanr(s["baseline"], s["actual"]).statistic
        verdict = "model" if rm > rb else "baseline"
        print(f"    {pos}: n={len(s):3d}  model {rm:.3f}   baseline {rb:.3f}   -> {verdict}")


if __name__ == "__main__":
    main()

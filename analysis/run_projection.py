"""Assemble the 2026/27 player frame and project expected points per gameweek."""
import json
import os
import numpy as np
import pandas as pd

from project import project_row, impute_from_price, DC_THRESH
from minutes import expected_minutes

PL25 = "data/2025-2026/By Tournament/Premier League"
POS_SHORT = {"Goalkeeper": "GKP", "Defender": "DEF",
             "Midfielder": "MID", "Forward": "FWD"}
# Shrinkage: how many league-average matches to blend into each rate.
SHRINK_RATE = 8.0
SHRINK_DC = 6.0


def prior_appearances() -> pd.DataFrame:
    """Exact PL appearance and 60-minute counts for 25/26, keyed on player_code."""
    frames = []
    for gw in sorted(os.listdir(PL25), key=lambda x: int(x[2:])):
        p = os.path.join(PL25, gw, "playermatchstats.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p, usecols=["player_id", "match_id", "minutes_played"]))
    pms = pd.concat(frames, ignore_index=True).drop_duplicates(["player_id", "match_id"])
    pms["minutes_played"] = pd.to_numeric(pms["minutes_played"], errors="coerce").fillna(0)
    pl = pd.read_csv("data/2025-2026/players.csv")
    pms = pms.merge(pl[["player_id", "player_code"]], on="player_id", how="left")
    played = pms[pms["minutes_played"] > 0]
    return played.groupby("player_code").agg(
        apps=("minutes_played", "size"),
        app60=("minutes_played", lambda s: int((s >= 60).sum())),
    ).reset_index()


def build_frame() -> pd.DataFrame:
    m = pd.read_csv("analysis/out_master.csv")
    m = m.merge(prior_appearances(), on="player_code", how="left")

    ts = pd.read_csv("analysis/out_team_strength.csv").set_index("team_s")
    t25 = pd.read_csv("data/2025-2026/teams.csv")
    code2short = t25.set_index("code")["short_name"].to_dict()
    m["old_team"] = m["p_team_code"].map(code2short)

    # Attack rating of the club the player played for last season. Used to
    # rescale their output into their new club's context.
    league_mean_atk = float(((ts["attack_H"] + ts["attack_A"]) / 2).mean())
    m["old_atk"] = m["old_team"].map(
        ((ts["attack_H"] + ts["attack_A"]) / 2).to_dict()).fillna(league_mean_atk)

    # Rate stats, shrunk toward the positional mean by sample size.
    m["n90"] = (m["p_mins"].fillna(0) / 90.0)
    for src, dst in [("p_xg90", "xg90"), ("p_xa90", "xa90"),
                     ("p_bps90", "bps90"), ("p_yc90", "yc90"),
                     ("p_saves90", "saves90")]:
        m[dst] = pd.to_numeric(m[src], errors="coerce")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        sel = m["pos"] == pos
        for c in ["xg90", "xa90", "bps90", "yc90", "saves90"]:
            mu = m.loc[sel & m["has_history"], c].mean()
            w = m.loc[sel, "n90"] / (m.loc[sel, "n90"] + SHRINK_RATE)
            m.loc[sel, c] = w * m.loc[sel, c].fillna(mu) + (1 - w) * mu

    # DefCon hit rate, shrunk toward the positional mean.
    m["dc_rate"] = pd.to_numeric(m["p_dc_rate"], errors="coerce")
    for pos in DC_THRESH:
        sel = m["pos"] == pos
        mu = m.loc[sel, "dc_rate"].mean()
        a = m.loc[sel, "p_apps"].fillna(0)
        w = a / (a + SHRINK_DC)
        m.loc[sel, "dc_rate"] = w * m.loc[sel, "dc_rate"].fillna(mu) + (1 - w) * mu
    m.loc[m["pos"] == "GKP", "dc_rate"] = 0.0

    m = impute_from_price(m, ["xg90", "xa90", "bps90", "yc90", "saves90", "dc_rate"])
    m["dc_rate"] = m["dc_rate"].clip(0, 0.95)

    # Minutes blend prior season, pre-season friendlies and price. See minutes.py
    # for why last season's minutes alone are not enough.
    m = expected_minutes(m)

    # Availability. Injured or suspended players are zeroed for GW1 purposes.
    chance = pd.to_numeric(m["chance_of_playing_next_round"], errors="coerce")
    avail = np.where(m["status"] == "a", 1.0, np.where(chance.notna(), chance / 100.0, 0.0))
    m["avail"] = avail
    return m


def project_gw(m: pd.DataFrame, fixtures: list, teams: dict, gw: int,
               ts: pd.DataFrame) -> pd.Series:
    """Expected points for every player in one gameweek."""
    opp, venue = {}, {}
    for f in fixtures:
        if f["event"] != gw:
            continue
        h, a = teams[f["team_h"]], teams[f["team_a"]]
        opp[h], venue[h] = a, "H"
        opp[a], venue[a] = h, "A"
    out = []
    for _, r in m.iterrows():
        t = r["team_s"]
        if t not in opp:
            out.append(0.0)
            continue
        v, o = venue[t], opp[t]
        atk = ts.loc[t, "attack_H" if v == "H" else "attack_A"]
        dfn = ts.loc[t, "defence_H" if v == "H" else "defence_A"]
        oa = ts.loc[o, "attack_A" if v == "H" else "attack_H"]
        od = ts.loc[o, "defence_A" if v == "H" else "defence_H"]
        out.append(project_row(r, atk, dfn, oa, od, v) * r["avail"])
    return pd.Series(out, index=m.index)


def main() -> None:
    m = build_frame()
    ts = pd.read_csv("analysis/out_team_strength.csv").set_index("team_s")
    boot = json.load(open(".research/bootstrap_static.json", encoding="utf-8"))
    fixtures = json.load(open(".research/fixtures.json", encoding="utf-8"))
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}

    for gw in range(1, 39):
        m[f"ep{gw}"] = project_gw(m, fixtures, teams, gw, ts)
    m["ep_gw1_6"] = m[[f"ep{g}" for g in range(1, 7)]].sum(axis=1)
    m["ep_season"] = m[[f"ep{g}" for g in range(1, 39)]].sum(axis=1)
    m["ep_per_m"] = m["ep_season"] / m["price"]
    m.to_csv("analysis/out_projections.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["web_name", "team_s", "pos", "price", "own", "ep1",
            "ep_gw1_6", "ep_season", "ep_per_m"]
    print(f"Projected {len(m)} players over 38 gameweeks.")
    print()
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        print("=" * 96)
        print(f"TOP 12 PROJECTED — {pos}")
        print("=" * 96)
        s = m[m["pos"] == pos].nlargest(12, "ep_season").copy()
        for c in ["ep1", "ep_gw1_6", "ep_season", "ep_per_m"]:
            s[c] = s[c].round(2)
        print(s[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()

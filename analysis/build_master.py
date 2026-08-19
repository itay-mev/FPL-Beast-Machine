"""Join 2025/26 Premier League performance onto the live 2026/27 player pool.

The 2026/27 season has not started, so every 26/27 stat column is zero. All
predictive signal has to come from prior seasons. This builds the bridge.

Join key is player_code. player_id is re-issued by FPL every season, so joining
on it across seasons silently pairs unrelated players.

Team is taken from the 2026/27 pool, never from 25/26, so summer transfers are
reflected. A player's rate stats travel with them, their team context does not.
"""
import json
import pandas as pd

BOOT = ".research/bootstrap_static.json"
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def load_pool() -> pd.DataFrame:
    d = json.load(open(BOOT, encoding="utf-8"))
    teams = {t["id"]: t["short_name"] for t in d["teams"]}
    e = pd.DataFrame(d["elements"])
    e["team_s"] = e["team"].map(teams)
    e["pos"] = e["element_type"].map(POS)
    e["price"] = e["now_cost"] / 10
    e["own"] = pd.to_numeric(e["selected_by_percent"], errors="coerce")
    keep = ["code", "web_name", "team_s", "pos", "price", "own", "status",
            "news", "chance_of_playing_next_round", "penalties_order",
            "corners_and_indirect_freekicks_order", "direct_freekicks_order"]
    return e[keep].rename(columns={"code": "player_code"})


def load_prior() -> pd.DataFrame:
    """Season-total FPL stats for 2025/26, keyed on the stable player_code."""
    ps = pd.read_csv("data/2025-2026/By Gameweek/GW38/playerstats.csv")
    pl = pd.read_csv("data/2025-2026/players.csv")
    ps = ps.merge(pl[["player_id", "player_code", "team_code"]],
                  left_on="id", right_on="player_id", how="left")

    num = ["total_points", "minutes", "starts", "goals_scored", "assists",
           "clean_sheets", "bonus", "bps", "expected_goals", "expected_assists",
           "expected_goals_conceded", "yellow_cards", "saves", "now_cost"]
    for c in num:
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)

    n90 = (ps["minutes"] / 90.0).clip(lower=0.5)
    out = pd.DataFrame({
        "player_code": ps["player_code"],
        "p_mins": ps["minutes"],
        "p_starts": ps["starts"],
        "p_pts": ps["total_points"],
        "p_ppg": ps["total_points"] / n90,
        "p_xg90": ps["expected_goals"] / n90,
        "p_xa90": ps["expected_assists"] / n90,
        "p_xgi90": (ps["expected_goals"] + ps["expected_assists"]) / n90,
        "p_xgc90": ps["expected_goals_conceded"] / n90,
        "p_bonus": ps["bonus"],
        "p_bps90": ps["bps"] / n90,
        "p_yc90": ps["yellow_cards"] / n90,
        "p_saves90": ps["saves"] / n90,
        "p_price_end": ps["now_cost"],
        "p_team_code": ps["team_code"],
    })
    return out.dropna(subset=["player_code"])


def main() -> None:
    pool = load_pool()
    prior = load_prior()
    dc = pd.read_csv("analysis/out_defcon_2025_26.csv")[
        ["player_code", "apps", "hits", "hit_rate", "actions_p90", "defcon_p90"]
    ].rename(columns={"apps": "p_apps", "hits": "p_dc_hits",
                      "hit_rate": "p_dc_rate", "actions_p90": "p_dc_actions90",
                      "defcon_p90": "p_dc_p90"})

    m = pool.merge(prior, on="player_code", how="left").merge(dc, on="player_code", how="left")
    m["has_history"] = m["p_mins"].notna() & (m["p_mins"] > 0)
    m.to_csv("analysis/out_master.csv", index=False)

    print(f"2026/27 pool: {len(pool)}")
    print(f"  with 25/26 PL history : {int(m['has_history'].sum())}")
    print(f"  no 25/26 PL history   : {int((~m['has_history']).sum())}")
    print()
    print("Players with no 25/26 PL minutes, by club (these need a separate prior):")
    nh = m[~m["has_history"]].groupby("team_s").size().sort_values(ascending=False)
    print(nh.to_string())
    print()
    pd.set_option("display.width", 250)
    print("=== BIGGEST PRICE-TO-PRIOR-OUTPUT MISMATCHES: cheap DEF with high DefCon ===")
    d = m[(m["pos"] == "DEF") & (m["p_apps"] >= 20) & (m["price"] <= 5.5)].copy()
    d["dc_rate_pct"] = (d["p_dc_rate"] * 100).round(1)
    print(d.nlargest(18, "p_dc_rate")[
        ["web_name", "team_s", "price", "own", "p_apps", "dc_rate_pct",
         "p_dc_actions90", "p_pts", "p_bonus"]].to_string(index=False))


if __name__ == "__main__":
    main()

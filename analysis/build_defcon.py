"""Per-player DefCon threshold-clearing rates from 2025/26 Premier League matches.

FPL 2026/27 awards 2 points, capped at 2 per match, when a player reaches:
  DEF          : 10  clearances + blocks + interceptions + tackles  (CBIT)
  MID and FWD  : 12  CBIT + recoveries                              (CBIRT)
  GKP          : not eligible

The award is a capped threshold, not a rate, so the quantity that matters is the
share of appearances in which a player clears the line. A player averaging 9.5
actions is worth far less than one averaging 10.5, and a spiky player who clears
the line often can beat a steadier player with a higher mean.

Action counts come from the source `defensive_contributions` column, which is
FPL's own per-match tally. Its position rule was verified against reconstructed
CBIT/CBIRT: defenders match CBIT, midfielders and forwards match CBIRT.

Scope: only "By Tournament/Premier League" is read. The "By Gameweek" folders
mix in cup and European fixtures, which would inflate every total.
Keying: results are keyed on player_code, which is stable across seasons.
player_id is re-issued each season and must never be used to join across them.
"""
import os
import pandas as pd

SEASON = "data/2025-2026"
PL = f"{SEASON}/By Tournament/Premier League"
CBIT = ["tackles", "clearances", "interceptions", "blocks"]
POS_SHORT = {"Goalkeeper": "GKP", "Defender": "DEF",
             "Midfielder": "MID", "Forward": "FWD"}
THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}


def load_pl_player_matches() -> pd.DataFrame:
    frames = []
    for gw in sorted(os.listdir(PL), key=lambda x: int(x[2:])):
        path = os.path.join(PL, gw, "playermatchstats.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["gw"] = int(gw[2:])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # One source fixture is filed under two gameweeks with the same match_id.
    return out.drop_duplicates(subset=["player_id", "match_id"])


def build() -> pd.DataFrame:
    pms = load_pl_player_matches()
    players = pd.read_csv(f"{SEASON}/players.csv")

    for col in CBIT + ["recoveries", "minutes_played", "defensive_contributions"]:
        pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

    pms = pms.merge(
        players[["player_id", "player_code", "web_name", "position", "team_code"]],
        on="player_id", how="left",
    )
    pms["pos"] = pms["position"].map(POS_SHORT)

    # Fall back to reconstruction on the ~2% of rows with no official tally.
    cbit = pms[CBIT].sum(axis=1)
    recon = cbit.where(pms["pos"] == "DEF", cbit + pms["recoveries"])
    pms["actions"] = pms["defensive_contributions"].where(
        pms["defensive_contributions"] > 0, recon)

    played = pms[(pms["minutes_played"] > 0) & pms["pos"].isin(THRESHOLD)].copy()
    played["hit"] = played["actions"] >= played["pos"].map(THRESHOLD)

    g = played.groupby(["player_code", "web_name", "pos"], as_index=False).agg(
        apps=("hit", "size"),
        mins=("minutes_played", "sum"),
        hits=("hit", "sum"),
        mean_actions=("actions", "mean"),
        total_actions=("actions", "sum"),
    )
    g["hit_rate"] = g["hits"] / g["apps"]
    g["actions_p90"] = g["total_actions"] / (g["mins"] / 90.0)
    g["defcon_pts"] = g["hits"] * 2
    g["defcon_p90"] = g["defcon_pts"] / (g["mins"] / 90.0)
    return g.sort_values("defcon_pts", ascending=False)


def main() -> None:
    g = build()
    g.to_csv("analysis/out_defcon_2025_26.csv", index=False)
    q = g[g["apps"] >= 15]

    pd.set_option("display.width", 250)
    cols = ["web_name", "pos", "apps", "mins", "hits", "hit_rate",
            "mean_actions", "actions_p90", "defcon_pts", "defcon_p90"]
    print(f"Players with >=15 PL appearances: {len(q)}")
    print()
    for pos in ["DEF", "MID", "FWD"]:
        print("=" * 110)
        print(f"TOP 20 DEFCON — {pos} (threshold {THRESHOLD[pos]} "
              f"{'CBIT' if pos == 'DEF' else 'CBIRT'})")
        print("=" * 110)
        s = q[q["pos"] == pos].head(20).copy()
        s["hit_rate"] = (s["hit_rate"] * 100).round(1)
        for c in ["mean_actions", "actions_p90", "defcon_p90"]:
            s[c] = s[c].round(2)
        print(s[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()

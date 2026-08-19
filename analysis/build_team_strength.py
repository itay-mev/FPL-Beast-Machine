"""Team attack and defence ratings for 2026/27 from prior-season match data.

FPL's own strength_attack_* and strength_defence_* fields are all zero before
the season starts, so the built-in FDR is unusable right now. This derives
ratings from actual match output instead.

Ratings are expressed as multipliers against the league average, split by venue:
  attack  > 1.0 means the team creates more than an average team
  defence > 1.0 means the team concedes more than an average team (worse)

Promoted clubs have no Premier League record. They are assigned an empirical
prior measured from how the previous cohort of promoted clubs actually
performed, rather than an arbitrary guess.
"""
import os
import pandas as pd

PL_25 = "data/2025-2026/By Tournament/Premier League"
SHRINK = 6.0  # matches of league-average prior mixed into every team's rating


def load_matches(base: str) -> pd.DataFrame:
    frames = []
    for gw in sorted(os.listdir(base), key=lambda x: int(x[2:])):
        p = os.path.join(base, gw, "matches.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    m = pd.concat(frames, ignore_index=True)
    m = m.drop_duplicates(subset=["match_id"])
    return m[m["finished"].astype(str).str.lower().isin(["true", "1"])]


def team_table(m: pd.DataFrame, names: dict) -> pd.DataFrame:
    """One row per team-match side, from both home and away perspectives."""
    home = pd.DataFrame({
        "team": m["home_team"], "opp": m["away_team"], "venue": "H",
        "gf": m["home_score"], "ga": m["away_score"],
        "xgf": pd.to_numeric(m["home_expected_goals_xg"], errors="coerce"),
        "xga": pd.to_numeric(m["away_expected_goals_xg"], errors="coerce"),
    })
    away = pd.DataFrame({
        "team": m["away_team"], "opp": m["home_team"], "venue": "A",
        "gf": m["away_score"], "ga": m["home_score"],
        "xgf": pd.to_numeric(m["away_expected_goals_xg"], errors="coerce"),
        "xga": pd.to_numeric(m["home_expected_goals_xg"], errors="coerce"),
    })
    t = pd.concat([home, away], ignore_index=True)
    t["team_s"] = t["team"].map(names)
    t["opp_s"] = t["opp"].map(names)
    # Blend xG with actual goals. xG is more stable, goals carry finishing skill.
    t["af"] = 0.7 * t["xgf"].fillna(t["gf"]) + 0.3 * t["gf"]
    t["aa"] = 0.7 * t["xga"].fillna(t["ga"]) + 0.3 * t["ga"]
    return t


def main() -> None:
    # matches.csv keys teams by the club `code`, not the season-local `id`,
    # despite what the repo README says. `code` is also stable across seasons.
    names25 = pd.read_csv("data/2025-2026/teams.csv").set_index("code")["short_name"].to_dict()
    m = load_matches(PL_25)
    t = team_table(m, names25)

    lg = {v: t[t["venue"] == v][["af", "aa"]].mean() for v in "HA"}
    print("2025/26 league averages (blended xG/goals per match):")
    for v in "HA":
        print(f"  {v}: created {lg[v]['af']:.2f}  conceded {lg[v]['aa']:.2f}")
    print()

    rows = []
    for (team, venue), g in t.groupby(["team_s", "venue"]):
        n = len(g)
        la, ld = lg[venue]["af"], lg[venue]["aa"]
        # Shrink small samples toward the league average.
        atk = (g["af"].sum() + SHRINK * la) / (n + SHRINK) / la
        dfn = (g["aa"].sum() + SHRINK * ld) / (n + SHRINK) / ld
        rows.append({"team_s": team, "venue": venue, "n": n,
                     "attack": atk, "defence": dfn})
    r = pd.DataFrame(rows).pivot(index="team_s", columns="venue",
                                 values=["attack", "defence"])
    r.columns = [f"{a}_{b}" for a, b in r.columns]
    r = r.reset_index()

    # Empirical promoted-club prior: which clubs were new to the PL in 25/26?
    names24 = pd.read_csv("data/2024-2025/teams/teams.csv")
    prev = set(names24["short_name"])
    promoted_25 = sorted(set(r["team_s"]) - prev)
    print(f"Clubs new to the PL in 2025/26 (used as the promoted prior): {promoted_25}")
    prior = r[r["team_s"].isin(promoted_25)][
        ["attack_H", "attack_A", "defence_H", "defence_A"]].mean()
    print("Their average rating in their promotion season:")
    print("  " + "  ".join(f"{k} {v:.3f}" for k, v in prior.items()))
    print()

    cur = pd.read_csv("data/2026-2027/teams.csv")["short_name"].tolist()
    out = []
    for team in cur:
        if team in set(r["team_s"]):
            row = r[r["team_s"] == team].iloc[0].to_dict()
            row["source"] = "25/26 record"
        else:
            row = {"team_s": team, **prior.to_dict(), "source": "promoted prior"}
        out.append(row)
    o = pd.DataFrame(out)
    o["net"] = (o["attack_H"] + o["attack_A"]) / 2 - (o["defence_H"] + o["defence_A"]) / 2
    o = o.sort_values("net", ascending=False)
    o.to_csv("analysis/out_team_strength.csv", index=False)

    pd.set_option("display.width", 200)
    show = o.copy()
    for c in ["attack_H", "attack_A", "defence_H", "defence_A", "net"]:
        show[c] = show[c].round(3)
    print("=== 2026/27 TEAM RATINGS (attack: higher=better, defence: LOWER=better) ===")
    print(show[["team_s", "attack_H", "attack_A", "defence_H", "defence_A",
                "net", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()

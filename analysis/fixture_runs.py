"""Fixture difficulty runs for 2026/27, using the model's own team ratings.

FPL's built-in FDR is a 2-5 integer and its underlying strength_attack and
strength_defence fields are all zero before the season starts, so it carries
little information right now. These runs use the attack and defence ratings
derived from actual match output instead, split by venue.

Two separate views are produced because they drive different decisions:
  attacking ease  how easy it is to score against the upcoming opponents
  defensive ease  how likely a clean sheet is against them
"""
import json

import numpy as np
import pandas as pd

LG_HOME_GF, LG_AWAY_GF = 1.54, 1.24


def build() -> tuple:
    ts = pd.read_csv("analysis/out_team_strength.csv").set_index("team_s")
    boot = json.load(open(".research/bootstrap_static.json", encoding="utf-8"))
    fixtures = json.load(open(".research/fixtures.json", encoding="utf-8"))
    names = {t["id"]: t["short_name"] for t in boot["teams"]}

    atk_rows, def_rows, opp_rows = {}, {}, {}
    for t in names.values():
        atk_rows[t] = {}
        def_rows[t] = {}
        opp_rows[t] = {}
    for f in fixtures:
        gw = f["event"]
        h, a = names[f["team_h"]], names[f["team_a"]]
        for team, opp, v in ((h, a, "H"), (a, h, "A")):
            ov = "A" if v == "H" else "H"
            # Expected goals this team scores, relative to an average team.
            atk_rows[team][gw] = ts.loc[team, f"attack_{v}"] * ts.loc[opp, f"defence_{ov}"]
            # Expected goals this team concedes.
            lg = LG_HOME_GF if v == "H" else LG_AWAY_GF
            def_rows[team][gw] = lg * ts.loc[team, f"defence_{v}"] * ts.loc[opp, f"attack_{ov}"]
            opp_rows[team][gw] = f"{opp}{'(h)' if v == 'H' else '(a)'}"
    return (pd.DataFrame(atk_rows).T.sort_index(axis=1),
            pd.DataFrame(def_rows).T.sort_index(axis=1),
            pd.DataFrame(opp_rows).T.sort_index(axis=1))


def main() -> None:
    atk, dfn, opp = build()
    atk.to_csv("analysis/out_fixture_attack.csv")
    dfn.to_csv("analysis/out_fixture_defence.csv")
    opp.to_csv("analysis/out_fixture_opponents.csv")

    pd.set_option("display.width", 250)
    for lo, hi, label in [(1, 6, "GW1-6"), (1, 10, "GW1-10"), (7, 16, "GW7-16")]:
        w = list(range(lo, hi + 1))
        print("=" * 88)
        print(f"FIXTURE RUNS {label}")
        print("=" * 88)
        a = atk[w].mean(axis=1).rename("attack_ease").sort_values(ascending=False)
        d = dfn[w].mean(axis=1).rename("xGA_per_game").sort_values()
        t = pd.concat([a, d], axis=1)
        t["clean_sheet_pct"] = (np.exp(-t["xGA_per_game"]) * 100).round(1)
        t = t.sort_values("attack_ease", ascending=False).round(3)
        print("Best ATTACKING runs (higher = easier to score):")
        print(t.head(8).to_string())
        print()
        print("Best DEFENSIVE runs (lower xGA = more clean sheets):")
        print(t.sort_values("xGA_per_game").head(8).to_string())
        print()

    print("=" * 88)
    print("OPPONENT SCHEDULE, GW1-8")
    print("=" * 88)
    print(opp[list(range(1, 9))].to_string())


if __name__ == "__main__":
    main()

"""Ablation study: test one model change at a time, then combine the winners.

Changing several things at once makes it impossible to attribute an improvement,
so each variant below alters exactly one knob from CURRENT. Combinations are
tested only after the individual effects are known.
"""
import pandas as pd

from harness import Variant, run

VARIANTS = [
    Variant("CURRENT (live model)", notes="full-season minutes, guessed bonus curve"),

    # Minutes is the dominant term in a season total, so it gets tested hardest.
    Variant("minutes: last 10 GWs", minutes_source="last10",
            notes="late-season role, ignores players who lost their place early"),
    Variant("minutes: blend full+late", minutes_source="blend",
            notes="recency-weighted role"),
    Variant("minutes: starter=85", starter_mins=85.0,
            notes="real starters play closer to 85 than 78"),
    Variant("minutes: starter=88", starter_mins=88.0),

    # Remove the hand-guessed bonus curve.
    Variant("bonus: fitted curve", bonus_curve="fitted",
            notes="power curve fit on real 25/26 bonus vs bps"),

    # Rate-stat handling.
    Variant("rates: combined xGI", xgi_combined=True,
            notes="xG+xA is more stable than each separately"),
    Variant("rates: heavier shrink", shrink_rate=14.0, shrink_dc=10.0,
            notes="more regression to positional mean"),
    Variant("rates: lighter shrink", shrink_rate=4.0, shrink_dc=3.0),

    # Price as a role signal.
    Variant("price weight 0.50", price_weight=0.50,
            notes="trust FPL's own valuation more"),
    Variant("price weight 0.15", price_weight=0.15),

    # Ensembling with the naive baseline.
    Variant("ensemble 25% baseline", ensemble_baseline=0.25),
    Variant("ensemble 40% baseline", ensemble_baseline=0.40),
    Variant("ensemble 60% baseline", ensemble_baseline=0.60),
]


def main() -> None:
    res = run(VARIANTS)
    pd.set_option("display.width", 250)
    cols = ["variant", "rho", "top50", "top50_mean_actual",
            "rho_GKP", "rho_DEF", "rho_MID", "rho_FWD", "notes"]
    show = res[cols].copy()
    for c in ["rho", "rho_GKP", "rho_DEF", "rho_MID", "rho_FWD"]:
        show[c] = show[c].round(4)
    show["top50_mean_actual"] = show["top50_mean_actual"].round(1)
    show = show.sort_values("rho", ascending=False)
    print("=" * 150)
    print("ABLATION: predict 2025/26 season points from 2024/25 only")
    print("=" * 150)
    print(show.to_string(index=False))
    res.to_csv("analysis/experiments/out_ablation.csv", index=False)


if __name__ == "__main__":
    main()

"""Variant-testing harness for the projection model.

The live model scores Spearman rho 0.553 against actual 2025/26 points, versus
0.528 for the naive "last season's points" baseline. That is a real but thin
edge. This harness makes model changes measurable rather than plausible.

Method: predict 2025/26 season totals using only 2024/25 data, then score
against what actually happened. Every variant is evaluated identically, so
differences are attributable to the change and not to the evaluation.

Season totals are dominated by minutes played, so the minutes model is the
highest-leverage thing to vary. A player who plays 3000 minutes outscores an
equally good player who plays 1500, regardless of rate stats.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

POS_SHORT = {"Goalkeeper": "GKP", "Defender": "DEF",
             "Midfielder": "MID", "Forward": "FWD"}
DC_T = {"DEF": 10, "MID": 12, "FWD": 12}
GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
LG_HOME_GF, LG_AWAY_GF = 1.54, 1.24
CBIT = ["tackles", "clearances", "interceptions", "blocks"]


@dataclass
class Variant:
    """One configuration of the projection model."""
    name: str
    # Which slice of the prior season informs the minutes model.
    minutes_source: str = "full"        # "full" | "last10" | "blend"
    starter_mins: float = 78.0
    # Bonus points curve.
    bonus_curve: str = "guess"          # "guess" | "fitted"
    # Blend the model with the naive baseline. 0.0 = pure model, 1.0 = pure baseline.
    ensemble_baseline: float = 0.0
    # Use combined expected goal involvements rather than separate xG and xA.
    xgi_combined: bool = False
    # Shrinkage strength for rate stats and for the DefCon hit rate.
    shrink_rate: float = 8.0
    shrink_dc: float = 6.0
    # Weight on price as a role signal in the minutes blend.
    price_weight: float = 0.30
    # How to fill rate stats for players with no prior-season record.
    impute_no_history: str = "posmean"   # "posmean" | "price"
    # How to rate clubs promoted into the target season.
    promoted: str = "prior"              # "prior" | "neutral" | "zero"
    # 0.0 pulls the promoted prior all the way to league average,
    # 1.0 applies the full weakest-three rating.
    promoted_severity: float = 1.0
    notes: str = ""


# ---------------------------------------------------------------- data loading

def load_prior_2425() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-player aggregates, per-gameweek player rows) for 2024/25."""
    pl = pd.read_csv("data/2024-2025/players/players.csv")
    base = "data/2024-2025/playermatchstats"
    frames = []
    for gw in range(1, 39):
        p = os.path.join(base, f"GW{gw}", "playermatchstats.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["gw"] = gw
            frames.append(df)
    pm = pd.concat(frames, ignore_index=True).drop_duplicates(["player_id", "match_id"])

    for c in ["minutes_played", "tackles", "clearances", "interceptions",
              "blocks", "recoveries", "saves", "yellow_cards"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce").fillna(0) if c in pm.columns else 0.0

    pm = pm.merge(pl[["player_id", "player_code", "position"]], on="player_id", how="left")
    pm["pos"] = pm["position"].map(POS_SHORT)
    cbit = pm[CBIT].sum(axis=1)
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
        last_gw=("gw", "max"),
    )
    agg["p_dc_rate"] = agg["p_dc_hits"] / agg["p_apps"]

    # Late-season slice. A player who lost their place by March should not be
    # projected as a starter, and a full-season average hides exactly that.
    late = played[played["gw"] >= 29]
    late_agg = late.groupby("player_code", as_index=False).agg(
        late_apps=("minutes_played", "size"),
        late_app60=("minutes_played", lambda s: int((s >= 60).sum())),
        late_mins=("minutes_played", "sum"),
    )
    agg = agg.merge(late_agg, on="player_code", how="left")

    ps = pd.read_csv("data/2024-2025/playerstats/playerstats.csv")
    ps = ps[ps["gw"] == 38].merge(
        pl[["player_id", "player_code", "team_code"]],
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
    return out, pm


def team_strength(season_dir: str, matches_rel: str, teams_rel: str) -> pd.DataFrame:
    m = pd.read_csv(f"{season_dir}/{matches_rel}").drop_duplicates("match_id")
    names = pd.read_csv(f"{season_dir}/{teams_rel}").set_index("code")["short_name"].to_dict()
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


def promoted_prior(ts: pd.DataFrame) -> pd.Series:
    """Rating to assign a club with no top-flight record in the source season.

    Newly promoted clubs cannot be rated from a league they did not play in.
    Historically they perform around the level of the existing bottom three, so
    the weakest three clubs by net rating are used as the prior. Treating such
    clubs as league-average would badly overrate them, and dropping them
    entirely discards real players who do score points.
    """
    net = ((ts["attack_H"] + ts["attack_A"]) / 2
           - (ts["defence_H"] + ts["defence_A"]) / 2)
    weakest = net.nsmallest(3).index
    return ts.loc[weakest].mean()


def extend_for_target(ts: pd.DataFrame, target_clubs: list,
                      severity: float = 1.0) -> pd.DataFrame:
    """Add rows for target-season clubs absent from the source-season ratings.

    Severity scales how harsh the prior is. The weakest three of any given
    season may have been unusually bad, so applying their rating at full
    strength can over-penalise the next promoted cohort. Severity blends
    between league average (0.0) and that full rating (1.0).
    """
    missing = [c for c in target_clubs if c not in ts.index]
    if not missing:
        return ts
    prior = promoted_prior(ts)
    blended = {k: 1.0 + severity * (v - 1.0) for k, v in prior.items()}
    add = pd.DataFrame([blended for _ in missing], index=missing)
    return pd.concat([ts, add])


def fit_bonus_curve() -> tuple[float, float]:
    """Fit bonus per 90 against BPS per 90 on real 2025/26 data.

    The live model uses a hand-guessed power curve. Fitting it removes that
    guess. Bonus is a rank-based prize within each match, so the relationship
    saturates rather than growing linearly.
    """
    ps = pd.read_csv("data/2025-2026/By Gameweek/GW38/playerstats.csv")
    for c in ["bonus", "bps", "minutes"]:
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)
    ok = ps[ps["minutes"] >= 900].copy()
    n90 = ok["minutes"] / 90.0
    ok["x"] = ok["bps"] / n90
    ok["y"] = ok["bonus"] / n90
    ok = ok[(ok["x"] > 0) & (ok["y"] > 0)]
    # Bin by BPS rate and fit the bin means. Fitting raw rows in log space lets
    # the mass of near-zero bonus players dominate and flattens the curve.
    ok["bin"] = pd.qcut(ok["x"], 12, duplicates="drop")
    g = ok.groupby("bin", observed=True)[["x", "y"]].mean()
    b, la = np.polyfit(np.log(g["x"]), np.log(g["y"]), 1)
    return float(np.exp(la)), float(b)


# ---------------------------------------------------------------- the model

def bonus_p90(bps90: np.ndarray, variant: Variant, fitted: tuple[float, float]) -> np.ndarray:
    if variant.bonus_curve == "fitted":
        a, b = fitted
        return np.clip(a * np.power(np.clip(bps90, 0.01, None), b), 0, 3.0)
    return np.clip(2.2 * np.power(bps90 / 32.0, 2.1), 0, 2.2)


def minutes_model(d: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    d = d.copy()
    apps = d["apps"].fillna(0)
    full_rate = (d["app60"].fillna(0) / 38.0).clip(0, 1)

    # Late-season start rate, over the 10 gameweeks that actually ended the season.
    late_n = d["late_apps"].fillna(0)
    late_rate = (d["late_app60"].fillna(0) / 10.0).clip(0, 1)

    if variant.minutes_source == "full":
        p_hist = full_rate
        hist_n = apps
    elif variant.minutes_source == "last10":
        p_hist = late_rate
        hist_n = late_n * 3.0   # a recent match is worth more than an old one
    else:  # blend
        w_late = late_n / (late_n + 4.0)
        p_hist = w_late * late_rate + (1 - w_late) * full_rate
        hist_n = apps + late_n

    # Price-implied start rate, fit within position on established starters.
    p_price = pd.Series(np.nan, index=d.index)
    for pos in d["pos"].dropna().unique():
        sel = d["pos"] == pos
        fit = d[sel & (apps >= 20)]
        if len(fit) < 8:
            p_price[sel] = 0.5
            continue
        yy = (fit["app60"] / 38.0).clip(0, 1)
        b, a = np.polyfit(fit["price"], yy, 1)
        p_price[sel] = np.clip(a + b * d.loc[sel, "price"], 0.05, 0.95)

    w_hist = 0.45 * (hist_n / (hist_n + 8.0))
    w_price = variant.price_weight
    w_none = 0.85 / (1.0 + hist_n / 2.5)
    tot = w_hist + w_price + w_none
    d["p_start"] = ((w_hist * p_hist.fillna(0)
                     + w_price * p_price.fillna(0.4)
                     + w_none * 0.10) / tot).clip(0, 0.95)

    # Goalkeeper is close to winner-takes-all within a club.
    gk = d["pos"] == "GKP"
    if gk.any():
        score = (d["own"].fillna(0) * 10.0 + apps * 0.5 + d["price"].fillna(4.0))
        d["_s"] = score
        rank = d[gk].groupby("team_s")["_s"].rank(ascending=False, method="first")
        d.loc[gk, "p_start"] = d.loc[gk, "p_start"] * rank.map(
            {1.0: 1.0, 2.0: 0.10}).fillna(0.03)
        d = d.drop(columns=["_s"])

    sub = ((apps - d["app60"].fillna(0)) / 38.0).clip(0, 0.5)
    d["p_app"] = (d["p_start"] + sub.fillna(0.1)).clip(0, 0.98)
    d["exp_mins"] = d["p_start"] * variant.starter_mins + (d["p_app"] - d["p_start"]) * 22.0
    return d


def project(d: pd.DataFrame, ts: pd.DataFrame, variant: Variant,
            fitted: tuple[float, float]) -> np.ndarray:
    """Season points against a neutral opponent, 19 home and 19 away."""
    pos = d["pos"].to_numpy()
    mins = d["exp_mins"].to_numpy()
    n90 = mins / 90.0
    p_app = d["p_app"].to_numpy()
    p60 = d["p_start"].to_numpy()

    atk_h = d["team_s"].map(ts["attack_H"]).fillna(1.0).to_numpy()
    atk_a = d["team_s"].map(ts["attack_A"]).fillna(1.0).to_numpy()
    dfn_h = d["team_s"].map(ts["defence_H"]).fillna(1.0).to_numpy()
    dfn_a = d["team_s"].map(ts["defence_A"]).fillna(1.0).to_numpy()
    old_atk = np.clip(d["old_atk"].to_numpy(), 0.35, None)

    goal_v = np.array([GOAL_PTS.get(p, 4) for p in pos], float)
    cs_v = np.array([CS_PTS.get(p, 0) for p in pos], float)
    is_gd = np.isin(pos, ["GKP", "DEF"]).astype(float)
    is_gk = (pos == "GKP").astype(float)

    xg90 = d["xg90"].to_numpy()
    xa90 = d["xa90"].to_numpy()
    if variant.xgi_combined:
        # Split combined involvements by each player's observed historical mix,
        # which is more stable than the two rates estimated separately.
        tot = np.clip(xg90 + xa90, 1e-6, None)
        share_g = np.where(tot > 0, xg90 / tot, 0.5)
        xgi = tot
        xg90, xa90 = xgi * share_g, xgi * (1 - share_g)

    total = np.zeros(len(d))
    for atk, dfn, lg in ((atk_h, dfn_h, LG_HOME_GF), (atk_a, dfn_a, LG_AWAY_GF)):
        ctx = atk / old_atk
        pts = xg90 * n90 * ctx * goal_v + xa90 * n90 * ctx * 3.0
        pts = pts + 2.0 * p60 + 1.0 * (p_app - p60)
        xga = lg * dfn
        pts = pts + np.exp(-np.clip(xga, 0.05, None)) * cs_v * p60
        pts = pts - 0.5 * xga * n90 * is_gd
        pts = pts + d["dc_rate"].to_numpy() * 2.0 * p_app
        pts = pts + is_gk * (d["saves90"].to_numpy() * n90) / 3.0
        pts = pts + bonus_p90(d["bps90"].to_numpy(), variant, fitted) * n90
        pts = pts - d["yc90"].to_numpy() * n90
        total = total + 19.0 * pts
    return total * d["avail"].to_numpy()


def build_eval_frame(prior: pd.DataFrame) -> pd.DataFrame:
    """The 2025/26 pool as it looked at the GW1 deadline, joined to 24/25 priors."""
    pool = pd.read_csv("data/2025-2026/By Gameweek/GW1/playerstats.csv")
    pl26 = pd.read_csv("data/2025-2026/players.csv")
    code2short = pd.read_csv("data/2025-2026/teams.csv").set_index("code")["short_name"].to_dict()
    pool = pool.merge(pl26[["player_id", "player_code", "position", "team_code"]],
                      left_on="id", right_on="player_id", how="left")
    pool["pos"] = pool["position"].map(POS_SHORT)
    pool["team_s"] = pool["team_code"].map(code2short)
    # The repo CSVs store now_cost already in millions. Only the bootstrap API
    # returns tenths. Dividing here would put every price at a tenth of its
    # real value and quietly unbind the budget constraint.
    pool["price"] = pd.to_numeric(pool["now_cost"], errors="coerce")
    pool["own"] = pd.to_numeric(pool["selected_by_percent"], errors="coerce").fillna(0)

    m = pool.merge(prior, on="player_code", how="left").copy()
    m["has_history"] = m["p_mins"].notna() & (m["p_mins"] > 0)
    old_names = pd.read_csv("data/2024-2025/teams/teams.csv").set_index("code")["short_name"].to_dict()
    m["old_team"] = m["p_team_code"].map(old_names)
    m["avail"] = np.where(m["status"] == "a", 1.0, 0.6)

    actual = pd.read_csv("data/2025-2026/By Gameweek/GW38/playerstats.csv").merge(
        pl26[["player_id", "player_code"]], left_on="id", right_on="player_id", how="left")
    act = actual.groupby("player_code")["total_points"].max().rename("actual")
    m = m.merge(act, on="player_code", how="left")
    m["actual"] = pd.to_numeric(m["actual"], errors="coerce").fillna(0)
    m["baseline"] = pd.to_numeric(m["prior_pts"], errors="coerce").fillna(0)
    return m


def apply_rates(m: pd.DataFrame, ts: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    m = m.copy()
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
            w = m.loc[sel, "n90"] / (m.loc[sel, "n90"] + variant.shrink_rate)
            m.loc[sel, c] = w * m.loc[sel, c].fillna(mu) + (1 - w) * mu
    m["dc_rate"] = pd.to_numeric(m["p_dc_rate"], errors="coerce")
    for pos in DC_T:
        sel = m["pos"] == pos
        mu = m.loc[sel, "dc_rate"].mean()
        a = m.loc[sel, "p_apps"].fillna(0)
        w = a / (a + variant.shrink_dc)
        m.loc[sel, "dc_rate"] = w * m.loc[sel, "dc_rate"].fillna(mu) + (1 - w) * mu
    m.loc[m["pos"] == "GKP", "dc_rate"] = 0.0
    m["dc_rate"] = m["dc_rate"].fillna(0).clip(0, 0.95)
    for c in ["xg90", "xa90", "bps90", "yc90", "saves90"]:
        m[c] = m[c].fillna(m[c].median())
    if variant.impute_no_history == "price":
        m = _impute_from_price(m, ["xg90", "xa90", "bps90", "yc90",
                                   "saves90", "dc_rate"])
    return m


def _impute_from_price(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Estimate rate stats for players with no record by regressing on price.

    The shrinkage step already leaves these players at the positional mean.
    This replaces that with a price-implied estimate, on the theory that FPL's
    own valuation carries information the positional mean does not.
    """
    out = df.copy()
    for pos in out["pos"].dropna().unique():
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


def score(m: pd.DataFrame, proj: np.ndarray, variant: Variant) -> dict:
    d = m.copy()
    d["proj_raw"] = proj
    if variant.ensemble_baseline > 0:
        # Rank-average the two, since their scales differ.
        r1 = pd.Series(d["proj_raw"]).rank(pct=True)
        r2 = pd.Series(d["baseline"]).rank(pct=True)
        w = variant.ensemble_baseline
        d["proj"] = (1 - w) * r1 + w * r2
    else:
        d["proj"] = d["proj_raw"]

    ev = d[d["pos"].notna()]
    best50 = set(ev.nlargest(50, "actual")["player_code"])
    top = ev.nlargest(50, "proj")
    out = {
        "rho": spearmanr(ev["proj"], ev["actual"]).statistic,
        "top50": len(set(top["player_code"]) & best50),
        "top50_mean_actual": top["actual"].mean(),
        "n": len(ev),
    }
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        s = ev[ev["pos"] == pos]
        out[f"rho_{pos}"] = spearmanr(s["proj"], s["actual"]).statistic
    return out


def run(variants: list[Variant], verbose: bool = True) -> pd.DataFrame:
    prior, _ = load_prior_2425()
    ts_raw = team_strength("data/2024-2025", "matches/matches.csv", "teams/teams.csv")
    base = build_eval_frame(prior)
    target_clubs = sorted(base["team_s"].dropna().unique())
    sev_cache = {}
    fitted = fit_bonus_curve()
    if verbose:
        print(f"fitted bonus curve: bonus_p90 = {fitted[0]:.4f} * bps90^{fitted[1]:.3f}")
        missing = [c for c in target_clubs if c not in ts_raw.index]
        pr = promoted_prior(ts_raw)
        print(f"clubs with no 24/25 top-flight record: {missing}")
        print("  promoted prior: " + "  ".join(f"{k} {v:.3f}" for k, v in pr.items()))
        print()

    rows = []
    for v in variants:
        if v.promoted == "prior":
            if v.promoted_severity not in sev_cache:
                sev_cache[v.promoted_severity] = extend_for_target(
                    ts_raw, target_clubs, v.promoted_severity)
            ts = sev_cache[v.promoted_severity]
        else:
            ts = extend_for_target(ts_raw, target_clubs, 0.0)
        m = apply_rates(base, ts, v)
        m = minutes_model(m, v)
        proj = project(m, ts, v, fitted)
        if v.promoted == "zero":
            proj = np.where(base["team_s"].isin(ts_raw.index).to_numpy(), proj, 0.0)
        r = score(m, proj, v)
        r["variant"] = v.name
        r["notes"] = v.notes
        rows.append(r)
    res = pd.DataFrame(rows)

    ev = base[base["pos"].notna()]
    best50 = set(ev.nlargest(50, "actual")["player_code"])
    btop = ev.nlargest(50, "baseline")
    bl = {
        "variant": "BASELINE last season pts", "notes": "reference",
        "rho": spearmanr(ev["baseline"], ev["actual"]).statistic,
        "top50": len(set(btop["player_code"]) & best50),
        "top50_mean_actual": btop["actual"].mean(), "n": len(ev),
    }
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        s = ev[ev["pos"] == pos]
        bl[f"rho_{pos}"] = spearmanr(s["baseline"], s["actual"]).statistic
    return pd.concat([pd.DataFrame([bl]), res], ignore_index=True)

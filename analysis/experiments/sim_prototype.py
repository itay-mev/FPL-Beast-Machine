"""GPU Monte Carlo prototype, used to choose the production simulation count.

Two jobs. It prototypes the two-level correlated sampler the engine needs, and
it measures how many simulations are actually required before the numbers stop
moving.

Choosing the count matters because P(win a 100-manager league) is a tail
probability near 1 percent, and tail probabilities need far more samples than
means. But there is a ceiling on useful precision: the projection itself is only
about 0.69 rank-correlated with reality, so reporting P(win) to three decimal
places would be false precision no matter how many draws are taken.

Memory design: season totals are accumulated gameweek by gameweek into an
[S, P] tensor rather than materialising [S, 38, P]. That keeps 200k simulations
inside a few hundred megabytes.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

GOAL_PTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
LG = {"H": 1.54, "A": 1.24}


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Install the Blackwell-compatible build with "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128")
    cap = torch.cuda.get_device_capability(0)
    if cap != (12, 0):
        raise RuntimeError(f"expected sm_120, got sm_{cap[0]}{cap[1]}")
    return torch.device("cuda")


class Sampler:
    """Two-level sampler: team outcomes first, then players within teams.

    Sampling players independently would understate variance badly, because an
    Arsenal clean sheet pays every Arsenal defender at once. A strategy stacking
    one defence would then look far safer than it is.
    """

    def __init__(self, frame: pd.DataFrame, ts: pd.DataFrame,
                 fixtures: list[dict], teams: dict, device: torch.device):
        self.dev = device
        self.f = frame.reset_index(drop=True)
        self.n = len(self.f)
        self.teams = sorted(ts.index)
        self.tidx = {t: i for i, t in enumerate(self.teams)}
        self.nt = len(self.teams)

        t = torch.tensor
        self.player_team = t([self.tidx.get(x, 0) for x in self.f["team_s"]],
                             device=device, dtype=torch.long)
        self.valid_team = t([x in self.tidx for x in self.f["team_s"]],
                            device=device, dtype=torch.bool)
        self.xg90 = t(self.f["xg90"].fillna(0).to_numpy(), device=device, dtype=torch.float32)
        self.xa90 = t(self.f["xa90"].fillna(0).to_numpy(), device=device, dtype=torch.float32)
        self.dc = t(self.f["dc_rate"].fillna(0).to_numpy(), device=device, dtype=torch.float32)
        self.p_start = t(self.f["p_start"].fillna(0).to_numpy(), device=device, dtype=torch.float32)
        self.p_app = t(self.f["p_app"].fillna(0).to_numpy(), device=device, dtype=torch.float32)
        self.mins = t(self.f["exp_mins"].fillna(0).to_numpy(), device=device, dtype=torch.float32)
        self.avail = t(self.f["avail"].fillna(0).to_numpy(), device=device, dtype=torch.float32)
        self.bonus90 = t(np.clip(2.2 * (self.f["bps90"].fillna(0).to_numpy() / 32.0) ** 2.1, 0, 2.2),
                         device=device, dtype=torch.float32)
        pos = self.f["pos"].fillna("MID")
        self.goal_v = t([GOAL_PTS.get(p, 4) for p in pos], device=device, dtype=torch.float32)
        self.cs_v = t([CS_PTS.get(p, 0) for p in pos], device=device, dtype=torch.float32)
        self.is_gd = t([1.0 if p in ("GKP", "DEF") else 0.0 for p in pos],
                       device=device, dtype=torch.float32)

        # Per-gameweek team expectations, precomputed once.
        self.lam_for = torch.zeros((38, self.nt), device=device)
        self.lam_against = torch.zeros((38, self.nt), device=device)
        self.plays = torch.zeros((38, self.nt), device=device)
        for fx in fixtures:
            gw = fx["event"]
            if gw is None or not (1 <= gw <= 38):
                continue
            h, a = teams[fx["team_h"]], teams[fx["team_a"]]
            for team, opp, v in ((h, a, "H"), (a, h, "A")):
                if team not in self.tidx or opp not in self.tidx:
                    continue
                ov = "A" if v == "H" else "H"
                i = self.tidx[team]
                self.lam_for[gw - 1, i] = float(
                    LG[v] * ts.loc[team, f"attack_{v}"] * ts.loc[opp, f"defence_{ov}"])
                self.lam_against[gw - 1, i] = float(
                    LG[v] * ts.loc[team, f"defence_{v}"] * ts.loc[opp, f"attack_{ov}"])
                self.plays[gw - 1, i] = 1.0

        # Each player's share of their club's expected goals, for allocation.
        share = torch.zeros(self.n, device=device)
        tot = torch.zeros(self.nt, device=device)
        w = self.xg90 * (self.mins / 90.0)
        tot.index_add_(0, self.player_team, w)
        share = w / torch.clamp(tot[self.player_team], min=1e-6)
        self.goal_share = share

    def season_totals(self, n_sims: int, seed: int,
                      chunk: int = 25000) -> torch.Tensor:
        """Season points per player, shape [n_sims, n_players]."""
        out = torch.zeros((n_sims, self.n), device=self.dev, dtype=torch.float16)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        for lo in range(0, n_sims, chunk):
            hi = min(lo + chunk, n_sims)
            s = hi - lo
            acc = torch.zeros((s, self.n), device=self.dev, dtype=torch.float32)
            for gw in range(38):
                playing = self.plays[gw][self.player_team] * self.valid_team.float()
                if float(playing.sum()) == 0:
                    continue
                lam_f = self.lam_for[gw].expand(s, self.nt)
                lam_a = self.lam_against[gw].expand(s, self.nt)
                team_gf = torch.poisson(lam_f, generator=g)
                team_ga = torch.poisson(lam_a, generator=g)

                # Team form scales every player at that club together, which is
                # what creates the correlation between team-mates.
                exp_gf = torch.clamp(self.lam_for[gw], min=1e-3)
                scale = team_gf / exp_gf.unsqueeze(0)
                pscale = scale[:, self.player_team]
                clean = (team_ga == 0).float()[:, self.player_team]
                conceded = team_ga.float()[:, self.player_team]

                started = torch.bernoulli(self.p_start.expand(s, self.n), generator=g)
                appeared = torch.bernoulli(self.p_app.expand(s, self.n), generator=g)
                appeared = torch.maximum(appeared, started)
                active = appeared * playing * self.avail

                n90 = (self.mins / 90.0).expand(s, self.n)
                goals = torch.poisson(
                    torch.clamp(self.xg90 * n90 * pscale, min=0, max=6), generator=g)
                assists = torch.poisson(
                    torch.clamp(self.xa90 * n90 * pscale, min=0, max=6), generator=g)
                defcon = torch.bernoulli(self.dc.expand(s, self.n), generator=g)

                pts = goals * self.goal_v + assists * 3.0
                pts = pts + 2.0 * started + 1.0 * (appeared - started)
                pts = pts + clean * self.cs_v * started
                pts = pts - 0.5 * conceded * self.is_gd * started
                pts = pts + defcon * 2.0
                pts = pts + self.bonus90 * n90
                acc = acc + pts * active
            out[lo:hi] = acc.half()
            del acc
        return out


def sample_field(frame: pd.DataFrame, totals: torch.Tensor, n_rivals: int,
                 seed: int, dev: torch.device) -> torch.Tensor:
    """Synthetic rivals drawn by real ownership, giving a mostly-template field."""
    own = torch.tensor(frame["own"].fillna(0.1).clip(lower=0.05).to_numpy(),
                       device=dev, dtype=torch.float32)
    pos = frame["pos"].fillna("MID").to_numpy()
    g = torch.Generator(device=dev).manual_seed(seed)
    S = totals.shape[0]
    scores = torch.zeros((S, n_rivals), device=dev, dtype=torch.float32)
    # Rivals are built position by position so every squad is legal.
    for p, k in [("GKP", 1), ("DEF", 4), ("MID", 4), ("FWD", 2)]:
        idx = torch.tensor(np.where(pos == p)[0], device=dev, dtype=torch.long)
        w = own[idx]
        pick = torch.multinomial(w.expand(n_rivals, len(idx)), k,
                                 replacement=False, generator=g)
        chosen = idx[pick]                       # [n_rivals, k]
        sel = totals[:, chosen.reshape(-1)].float().reshape(S, n_rivals, k)
        scores += sel.sum(dim=2)
    # Captain doubling, approximated as the rival's best scorer.
    return scores * 1.12


def main() -> None:
    dev = require_cuda()
    frame = pd.read_csv("analysis/out_projections.csv")
    frame = frame[frame["pos"].notna()].reset_index(drop=True)
    ts = pd.read_csv("analysis/out_team_strength.csv").set_index("team_s")
    boot = json.load(open(".research/bootstrap_static.json", encoding="utf-8"))
    fixtures = json.load(open(".research/fixtures.json", encoding="utf-8"))
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}

    smp = Sampler(frame, ts, fixtures, teams, dev)
    print(f"players {smp.n}, teams {smp.nt}")
    print()

    # A representative squad to measure against: the current optimiser output.
    squad_names = ["Raya", "Gabriel", "Virgil", "Senesi", "Thiaw", "Anderson",
                   "Foden", "Szoboszlai", "Haaland", "João Pedro", "Welbeck",
                   "Dubravka", "Mitchell", "Ampadu", "Gomez"]
    sq = frame.index[frame["web_name"].isin(squad_names)].to_numpy()[:15]
    sq_t = torch.tensor(sq, device=dev, dtype=torch.long)

    print("=" * 96)
    print("CONVERGENCE: how P(win) and mean settle as simulation count rises")
    print("=" * 96)
    print(f"{'sims':>8} {'seeds':>6} {'mean pts':>10} {'sd(mean)':>9} "
          f"{'P(win) %':>9} {'sd P(win)':>10} {'time s':>8} {'VRAM GiB':>9}")
    rows = []
    for n_sims in [1000, 5000, 10000, 25000, 50000, 100000, 200000]:
        means, pwins, t_tot = [], [], 0.0
        seeds = 5 if n_sims <= 50000 else 3
        torch.cuda.reset_peak_memory_stats()
        for k in range(seeds):
            t0 = time.perf_counter()
            tot = smp.season_totals(n_sims, seed=1000 + k)
            own = tot[:, sq_t].float().sum(dim=1) * 1.12
            field = sample_field(frame, tot, 99, seed=5000 + k, dev=dev)
            torch.cuda.synchronize()
            t_tot += time.perf_counter() - t0
            means.append(float(own.mean()))
            pwins.append(float((own > field.max(dim=1).values).float().mean()) * 100)
            del tot, own, field
            torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append(dict(sims=n_sims, mean=np.mean(means), sd_mean=np.std(means),
                         pwin=np.mean(pwins), sd_pwin=np.std(pwins),
                         secs=t_tot / seeds, vram=peak))
        r = rows[-1]
        print(f"{n_sims:>8} {seeds:>6} {r['mean']:>10.1f} {r['sd_mean']:>9.2f} "
              f"{r['pwin']:>9.3f} {r['sd_pwin']:>10.4f} {r['secs']:>8.2f} {peak:>9.2f}")
    pd.DataFrame(rows).to_csv("analysis/experiments/out_convergence.csv", index=False)


if __name__ == "__main__":
    main()

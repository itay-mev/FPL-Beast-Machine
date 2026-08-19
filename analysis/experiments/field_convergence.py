"""Choose the production simulation count, with the field resampled per simulation.

An earlier version of this experiment drew the 99 rivals once and reused them
across every simulation. P(win) then had an irreducible standard deviation of
about 5.7 percentage points no matter how many simulations were run, because the
dominant uncertainty was which rivals happened to be drawn rather than how the
season played out. Strategy comparison would have been meaningless.

Drawing a fresh field for every simulation integrates over both sources of
uncertainty, so the estimate converges.

Rivals are sampled without replacement, weighted by ownership, using the
Gumbel top-k trick. Building an explicit multinomial over every simulation and
rival would need gigabytes. Adding Gumbel noise to log weights and taking the
top k is equivalent and needs one tensor per block.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from sim_prototype import Sampler, require_cuda

RIVALS = 99
SQUAD_SHAPE = [("GKP", 1), ("DEF", 4), ("MID", 4), ("FWD", 2)]


def gumbel_topk(logw: torch.Tensor, k: int, shape: tuple,
                g: torch.Generator) -> torch.Tensor:
    """Weighted sampling without replacement.

    Adding Gumbel noise to log weights and taking the top k is equivalent to
    sequential weighted sampling without replacement, and needs no loop.
    """
    u = torch.rand(shape + (logw.numel(),), device=logw.device, generator=g)
    keys = logw + (-torch.log(-torch.log(u.clamp_(1e-12, 1 - 1e-12))))
    return keys.topk(k, dim=-1).indices


def field_max(totals: torch.Tensor, pos_idx: dict, logw: dict,
              g: torch.Generator, rival_block: int = 11) -> torch.Tensor:
    """Best rival score per simulation, with a fresh field for each simulation.

    Rivals are processed in blocks so the gathered tensor stays small.
    """
    n = totals.shape[0]
    best = torch.full((n,), -1e9, device=totals.device)
    done = 0
    while done < RIVALS:
        b = min(rival_block, RIVALS - done)
        score = torch.zeros((n, b), device=totals.device)
        top = torch.zeros((n, b), device=totals.device)
        for p, k in SQUAD_SHAPE:
            idx = pos_idx[p]
            pick = gumbel_topk(logw[p], k, (n, b), g)      # [n, b, k]
            chosen = idx[pick]
            v = totals.gather(1, chosen.reshape(n, -1).clamp_(0, totals.shape[1] - 1)
                              ).float().reshape(n, b, k)
            score += v.sum(dim=2)
            top = torch.maximum(top, v.max(dim=2).values)
        best = torch.maximum(best, (score + top).max(dim=1).values)
        done += b
    return best


def main() -> None:
    dev = require_cuda()
    frame = pd.read_csv("analysis/out_projections.csv")
    frame = frame[frame["pos"].notna()].reset_index(drop=True)
    ts = pd.read_csv("analysis/out_team_strength.csv").set_index("team_s")
    boot = json.load(open(".research/bootstrap_static.json", encoding="utf-8"))
    fixtures = json.load(open(".research/fixtures.json", encoding="utf-8"))
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    smp = Sampler(frame, ts, fixtures, teams, dev)

    own = frame["own"].fillna(0.1).clip(lower=0.05).to_numpy()
    posv = frame["pos"].fillna("MID").to_numpy()
    K = 2.5
    pos_idx, logw = {}, {}
    for p, _ in SQUAD_SHAPE:
        ii = np.where(posv == p)[0]
        pos_idx[p] = torch.tensor(ii, device=dev, dtype=torch.long)
        logw[p] = torch.log(torch.tensor(own[ii] ** K, device=dev, dtype=torch.float32))

    names = ["Raya", "Gabriel", "Virgil", "Senesi", "Thiaw", "Anderson", "Foden",
             "Szoboszlai", "Haaland", "João Pedro", "Welbeck", "Dubravka",
             "Mitchell", "Ampadu", "Gomez"]
    sq = torch.tensor(frame.index[frame["web_name"].isin(names)].to_numpy()[:15],
                      device=dev, dtype=torch.long)

    def one(n: int, seed: int) -> tuple:
        tot = smp.season_totals(n, seed=seed)
        s = tot[:, sq].float()
        top = s.topk(11, dim=1).values
        ours = top.sum(dim=1) + top[:, 0]
        g = torch.Generator(device=dev).manual_seed(seed + 991)
        rmax = field_max(tot, pos_idx, logw, g)
        pw = float((ours > rmax).float().mean()) * 100
        res = (float(ours.mean()), float(rmax.mean()), pw)
        del tot, s, top, ours, rmax
        torch.cuda.empty_cache()
        return res

    print(f"field own^{K}, {RIVALS} rivals RESAMPLED PER SIMULATION")
    print("real 25/26 average manager total = 1895 (reference)")
    print()
    hdr = (f"{'sims':>8} {'seeds':>6} {'ours':>7} {'best rival':>11} "
           f"{'P(win) %':>9} {'sd':>8} {'95% CI +/-':>11} {'time s':>8} {'VRAM GiB':>9}")
    print(hdr)
    rows = []
    for n in [1000, 2500, 5000, 10000, 25000, 50000, 100000]:
        seeds = 6 if n <= 25000 else 4
        o, r, w, t = [], [], [], 0.0
        torch.cuda.reset_peak_memory_stats()
        for k in range(seeds):
            t0 = time.perf_counter()
            a, b, c = one(n, 3000 + k * 17)
            t += time.perf_counter() - t0
            o.append(a)
            r.append(b)
            w.append(c)
        sd = float(np.std(w, ddof=1))
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append(dict(sims=n, ours=np.mean(o), rival=np.mean(r),
                         pwin=np.mean(w), sd=sd, ci=1.96 * sd,
                         secs=t / seeds, vram=peak))
        print(f"{n:>8} {seeds:>6} {np.mean(o):>7.0f} {np.mean(r):>11.0f} "
              f"{np.mean(w):>9.3f} {sd:>8.4f} {1.96 * sd:>11.4f} "
              f"{t / seeds:>8.2f} {peak:>9.2f}")
    pd.DataFrame(rows).to_csv("analysis/experiments/out_convergence.csv", index=False)


if __name__ == "__main__":
    main()

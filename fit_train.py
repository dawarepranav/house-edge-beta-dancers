"""Fit params.json from public/train only: python fit_train.py --data public --out params.json

Uses train queries and histories. Validation targets are never read. Structure (lag pairs, size
lags and the cross-machine lag) is discovered first, then parameters are fitted by hard EM with a
logistic M-step. Deterministic: no random numbers are used.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import he_model as H

# Settings chosen on validation (everything else below is fitted on train histories).
SETTINGS = {"rho": 0.01, "re_var": 0.05, "cond_prior": [0.5, 0.25, 0.25], "iters": 2,
            "lag_tie": 0.98, "ramp": 20, "ramp_common": 0, "ramp_levels": 4,
            "cond_prior_short": [0.98, 0.01, 0.01], "short_len": 100}
ALL_PAIRS = [(i, j) for i in range(1, 9) for j in range(i + 1, 9)]
CROSS_LAGS = range(1, 9)
DISCOVERY_END = 100  # flips in this data happen later, so early ticks show the original rule
EM_ROUNDS = 6
FLIP_GUARD = 25  # ticks around a flip are skipped in the M-step because the change can be gradual


def long_episodes(episodes, history):
    return {e: rs for e, rs in episodes.items()
            if len(history[H.machine_key(rs[0])]) > 100}


def discover(episodes, history):
    win, size = Counter(), Counter()
    for rs in episodes.values():
        for q in rs:
            s = history[H.machine_key(q)]
            best, top = None, -1.0
            for pair in ALL_PAIRS:
                a = n = 0
                for t, v in s.items():
                    x, y = s.get(t - pair[0]), s.get(t - pair[1])
                    if t < DISCOVERY_END and x is not None and y is not None:
                        a += (v > 0) == ((x > 0) == (y > 0))
                        n += 1
                score = abs(a - n / 2) / math.sqrt(n / 4) if n else 0
                if score > top:
                    best, top = pair, score
            win[best] += 1
            bestl, topl = None, -1.0
            wins = [(t, v) for t, v in s.items() if v > 0]
            for lag in range(1, 9):
                c = [[0, 0], [0, 0]]
                for t, v in wins:
                    b = s.get(t - lag)
                    if b is not None:
                        c[b >= 2][v == 1] += 1
                score = 0.0
                n = sum(map(sum, c))
                for i in (0, 1):
                    for j in (0, 1):
                        e = sum(c[i]) * (c[0][j] + c[1][j]) / max(n, 1)
                        if e:
                            score += (c[i][j] - e) ** 2 / e
                if score > topl:
                    bestl, topl = lag, score
            size[bestl] += 1
    pairs = [p for p, _ in win.most_common(3)]
    # size lags listed in the order of their pairs' first lags, so the tie prior lines up
    lags = [l for l, _ in size.most_common(3)]
    lags = sorted(lags, key=lambda l: [p[0] for p in pairs].index(l) if l in [p[0] for p in pairs] else 9)
    return pairs, lags, win, size


def initial(episodes, history, pairs, lags):
    counts = {ty: [1.0] * 4 for ty in H.TYPES}
    for rs in episodes.values():
        for q in rs:
            for v in history[H.machine_key(q)].values():
                if v > 0:
                    counts[q["machine_type"]][H.PAIR_INDEX[v] - 1] += 1
    sizes = {}
    for ty, c in counts.items():
        z = sum(c)
        d = [v / z for v in c]
        hi = [d[0] * 1.3] + d[1:]
        lo = [d[0] * 0.7] + d[1:]
        sizes[ty] = {"hi": [v / sum(hi) for v in hi], "lo": [v / sum(lo) for v in lo]}
    P = dict(base={t: 0.0 for t in H.TYPES}, beta={t: 0.0 for t in H.TYPES},
             gamma={t: 0.0 for t in H.TYPES}, K=2.0, delta=0.3, sizes=sizes, pairs=pairs,
             size_lags=lags, taus=list(range(88, 258, 2)), cross_lag=0, cross_lam=0.0, cross_c0=0.0)
    P.update(SETTINGS)
    return P


def solve(A, b):
    n = len(b)
    M = [row[:] + [v] for row, v in zip(A, b)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(M[k][i]))
        M[i], M[p] = M[p], M[i]
        for k in range(n):
            if k != i:
                f = M[k][i] / M[i][i]
                for j in range(i, n + 1):
                    M[k][j] -= f * M[i][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def logistic(rows_, th, iters=15):
    """Newton logistic fit on aggregated rows (x, wins, n). Returns theta and log-likelihood."""
    k = len(th)
    for _ in range(iters):
        g = [0.0] * k
        A = [[1e-6 if i == j else 0.0 for j in range(k)] for i in range(k)]
        for x, w, n in rows_:
            p = H.sig(sum(a * b for a, b in zip(th, x)))
            for i in range(k):
                if x[i]:
                    g[i] += x[i] * (w - n * p)
                    for j in range(k):
                        if x[j]:
                            A[i][j] += n * p * (1 - p) * x[i] * x[j]
        d = solve(A, g)
        th = [a + b for a, b in zip(th, d)]
        if max(abs(v) for v in d) < 1e-7:
            break
    ll = 0.0
    for x, w, n in rows_:
        p = min(max(H.sig(sum(a * b for a, b in zip(th, x))), 1e-12), 1 - 1e-12)
        ll += w * math.log(p) + (n - w) * math.log(1 - p)
    return th, ll


def design(key, with_cross):
    ty, sg, z, xs, hs, c = key
    v = [0.0] * (12 if with_cross else 11)
    i = H.TYPES.index(ty)
    v[3 * i], v[3 * i + 1], v[3 * i + 2] = 1.0, float(sg), float(z)
    v[9], v[10] = float(xs), float(hs)
    if with_cross:
        v[11] = c
    return v


def m_step(P, cells, sizes):
    th = [0.0] * 12
    for i, t in enumerate(H.TYPES):
        th[3 * i], th[3 * i + 1], th[3 * i + 2] = P["base"][t], P["beta"][t], P["gamma"][t]
    th[9], th[10], th[11] = P["K"], P["delta"], P["cross_lam"]
    use_cross = P["cross_lag"] > 0
    lag = P["cross_lag"]
    rows_ = [(design(k, use_cross), v[0], v[1]) for k, v in cells[lag].items() if v[1] > 0]
    th, _ = logistic(rows_, th[:12 if use_cross else 11])
    for i, t in enumerate(H.TYPES):
        P["base"][t], P["beta"][t], P["gamma"][t] = th[3 * i], th[3 * i + 1], th[3 * i + 2]
    P["K"], P["delta"] = th[9], abs(th[10])
    if use_cross:
        P["cross_lam"] = th[11]
    for ty in H.TYPES:
        for st in ("hi", "lo"):
            c = sizes[(ty, st)]
            z = sum(c) + 4 * 0.5
            P["sizes"][ty][st] = [(v + 0.5) / z for v in c]


def choose_cross_lag(P, cells):
    """Compare single-lag cross terms by train log-likelihood gain over no cross term."""
    th0 = [0.0] * 11
    for i, t in enumerate(H.TYPES):
        th0[3 * i], th0[3 * i + 1], th0[3 * i + 2] = P["base"][t], P["beta"][t], P["gamma"][t]
    th0[9], th0[10] = P["K"], P["delta"]
    gains = {}
    for lag in CROSS_LAGS:
        # same rows with and without the cross term, so the gain is comparable across lags
        _, ll0 = logistic([(design(k, False), v[0], v[1]) for k, v in cells[lag].items() if v[1] > 0], th0)
        th, ll = logistic([(design(k, True), v[0], v[1]) for k, v in cells[lag].items() if v[1] > 0], th0 + [0.0])
        gains[lag] = (round(ll - ll0, 2), round(th[11], 4))
    return gains


def e_step(episodes, history, P, lags):
    """Hard E-step along each machine's MAP rule; cells[lag][key] = [wins, n] with soft hot/cold."""
    cells = {lag: defaultdict(lambda: [0.0, 0.0]) for lag in lags}
    sizes = defaultdict(lambda: [0.0] * 4)
    Q = dict(P, iters=2, re_var=0.0)
    J = H.widths(Q)[2]
    for e in sorted(episodes):
        rs = episodes[e]
        xo = H.cross_offsets(rs, history, Q)
        counts = {lag: H.cross_counts(rs, history, lag) for lag in lags if lag}
        machines = [H.Machine(q, history[H.machine_key(q)], Q, None, None, xo) for q in rs]
        W, per_tau = H.condition_posteriors(machines)
        _, offsets = H.episode_hot(machines, W, per_tau)
        machines = [H.Machine(q, history[H.machine_key(q)], Q, offsets, None, xo) for q in rs]
        W, per_tau = H.condition_posteriors(machines)
        _, offsets = H.episode_hot(machines, W, per_tau)
        for m in machines:
            h, tau, wd = H.best_path(m, W, per_tau)
            pi, s, li, z = h
            sg = m.q["signal"]
            key_m = H.machine_key(m.q)
            for t in m.ts:
                x = m.bits(t, Q["pairs"][pi])
                b = m.zbit(t, Q["size_lags"][li])
                if x is None or b is None or (tau is not None and tau - FLIP_GUARD <= t <= tau + wd + FLIP_GUARD):
                    continue
                zv = b * z
                S = s * H.mult_at(t, tau, wd, J)
                won = m.s[t] > 0
                pH = (offsets.get(t, 0.0) / Q["delta"] + 1) / 2
                for lag in lags:
                    c = 0.0
                    if lag:
                        c = counts[lag][key_m].get(t)
                        if c is None:
                            continue
                        c -= Q["cross_c0"] if lag == Q["cross_lag"] else 0.0
                    for hs, w in ((1, pH), (-1, 1 - pH)):
                        cell = cells[lag][(m.ty, sg, zv, S * x, hs, round(c, 3))]
                        cell[0] += w * won
                        cell[1] += w
                if won:
                    sizes[(m.ty, "hi" if zv > 0 else "lo")][H.PAIR_INDEX[m.s[t]] - 1] += 1
    return cells, sizes


def mean_count(episodes, history, lag):
    tot = n = 0.0
    for rs in episodes.values():
        for d in H.cross_counts(rs, history, lag).values():
            for c in d.values():
                if c is not None:
                    tot += c
                    n += 1
    return tot / n


def fit(data):
    queries, history, episodes = H.load_split(Path(data) / "train")
    use = long_episodes(episodes, history)
    pairs, lags, win, size = discover(use, history)
    print("win-rule pair counts:", win.most_common(8))
    print("size-rule lag counts:", size.most_common(8))
    P = initial(use, history, pairs, lags)
    for r in range(EM_ROUNDS):
        search = r == 2  # pick the cross-machine lag once the rule parameters have settled
        cells, sizes = e_step(use, history, P, list(CROSS_LAGS) if search else [P["cross_lag"]])
        if search:
            gains = choose_cross_lag(P, cells)
            print("cross-lag log-lik gains (gain, coef):", gains, flush=True)
            best = max(gains, key=lambda k: gains[k][0])
            P["cross_lag"] = best
            P["cross_c0"] = round(mean_count(use, history, best), 4)
            cells, sizes = e_step(use, history, P, [best])
        m_step(P, cells, sizes)
        print("round", r, "K", round(P["K"], 3), "delta", round(P["delta"], 3), "cross", P["cross_lag"],
              round(P["cross_lam"], 4), "base", {t: round(v, 3) for t, v in P["base"].items()}, flush=True)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to the round's public folder")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(fit(a.data), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

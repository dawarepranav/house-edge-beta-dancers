"""House Edge joint Bayesian inversion of the slot simulator. Standard library only.

Per machine:   win rule  X(t) = XNOR(win[t-i], win[t-j]), (i,j) in PAIRS, sign s,
               size rule Z(t) = sign z * (+1 if payout[t-L] >= 2 else -1), L in SIZE_LAGS
               (L is usually the first lag of the win pair; prior weight lag_tie)
Per episode:   hot/cold H(t) = +/-1, a two-state Markov chain.
               cross term C(t) = how many OTHER machines in the episode won at t - cross_lag.
logit P(win) = base[type] + K*s*m(t)*X + beta*signal + gamma*Z + delta*H + lam*(C - c0)
m(t) is +1 before a flip and -1 after it. A per-machine flip (drift) reverses gradually:
m(t) falls linearly over `ramp` ticks (piecewise constant in `ramp_levels` steps). A common
flip (shared change) uses width `ramp_common` (0 = a sharp step).
Three episode explanations are averaged by evidence: no flip, per-machine flip
(drift), single common flip tick (shared change). Episodes with sparse (short) histories use
their own prior, `cond_prior_short`, detected from the history length, not from labels.
"""
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
TYPES = ("steady", "balanced", "jackpot")
PAIR_INDEX = {p: i for i, p in enumerate(PAYOUTS)}


def lse(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def machine_key(q):
    """Histories are grouped by both IDs, as the participant guide requires."""
    return (q["episode_id"], q["machine_id"])


def load_split(root):
    root = Path(root)
    with open(root / "queries.csv", newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    for q in queries:
        q["signal"] = int(q["signal"])
        q["cutoff"] = int(q["cutoff"])
    history = defaultdict(dict)
    with open(root / "history.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            history[machine_key(r)][int(r["tick"])] = int(r["payout"])
    episodes = defaultdict(list)
    for q in queries:
        episodes[q["episode_id"]].append(q)
    return queries, history, episodes


def ramp_levels(J):
    return [round(1 - 2 * (j + .5) / J, 6) for j in range(J)]


def ramp_bounds(tau, width, J):
    return [tau + int(round(j * width / J)) for j in range(J + 1)]


def mult_at(t, tau, width, J):
    """Rule multiplier m(t) for a flip that starts at tau and takes `width` ticks."""
    if tau is None or t < tau:
        return 1.0
    if width <= 0:
        return -1.0
    bd = ramp_bounds(tau, width, J)
    if t >= bd[-1]:
        return -1.0
    lv = ramp_levels(J)
    for j in range(J):
        if bd[j] <= t < bd[j + 1]:
            return lv[j]
    return -1.0


def widths(P):
    return P.get("ramp", 0), P.get("ramp_common", 0), P.get("ramp_levels", 4)


class Machine:
    """All per-tick tables for one machine; hypothesis = (pair, s, lag, z)."""

    def __init__(self, q, hist, P, offsets=None, u=None, xo=None):
        self.q, self.s, self.P = q, hist, P
        self.ty, self.cut = q["machine_type"], q["cutoff"]
        self.u = (u or {}).get(machine_key(q), 0.0)
        self.xo = (xo or {}).get(machine_key(q), {})
        self.ticks = sorted(t for t in hist if t < self.cut)
        tie = P.get("lag_tie", 0.0)
        self.hyp = [(p, s, l, z) for p in range(len(P["pairs"])) for s in (1, -1)
                    for l in range(len(P["size_lags"])) for z in (1, -1)
                    if tie < 1 or P["size_lags"][l] == P["pairs"][p][0]]
        self.build(offsets or {})

    def bits(self, t, pair):
        a, b = self.s.get(t - pair[0]), self.s.get(t - pair[1])
        return None if a is None or b is None else (1 if (a > 0) == (b > 0) else -1)

    def zbit(self, t, lag):
        v = self.s.get(t - lag)
        return None if v is None else (1 if v >= 2 else -1)

    def base(self):
        P = self.P
        return P["base"][self.ty] + P["beta"][self.ty] * self.q["signal"] + self.u

    def tick_ll(self, t, won, x, z, S, o, H=None):
        P = self.P
        g = P["gamma"][self.ty]
        z_vals = ((z, 1.0),) if z is not None else ((1, .5), (-1, .5))
        x_vals = ((x, 1.0),) if x is not None else ((1, .5), (-1, .5))
        tot = 0.0
        b = self.base() + o + self.xo.get(t, 0.0)
        for zv, zw in z_vals:
            for xv, xw in x_vals:
                eta = b + P["K"] * S * xv + g * zv
                pw = sig(eta)
                tot += zw * xw * (pw if won else 1 - pw)
        return math.log(max(tot, 1e-12))

    def _flip_ll(self, A, s, n, width, J):
        """Log-lik for every tau: +s before, ramp levels during, -s after."""
        out = []
        lv = ramp_levels(J)
        for tau in self.taus:
            if width <= 0:
                k = bisect.bisect_left(self.ts, tau)
                out.append(A[s][k] + A[-s][n] - A[-s][k])
                continue
            ks = [bisect.bisect_left(self.ts, b) for b in ramp_bounds(tau, width, J)]
            v = A[s][ks[0]] + A[-s][n] - A[-s][ks[-1]]
            for j in range(J):
                if ks[j + 1] > ks[j]:
                    key = round(s * lv[j], 6)
                    v += A[key][ks[j + 1]] - A[key][ks[j]]
            out.append(v)
        return out

    def build(self, offsets):
        """pre[(pair,lag,z)][S] = prefix of log-lik with rule multiplier S."""
        P = self.P
        self.ts = self.ticks
        self.taus = P["taus"]
        w_m, w_c, J = widths(P)
        vals = {1.0, -1.0}
        if w_m > 0 or w_c > 0:
            for l in ramp_levels(J):
                vals |= {round(l, 6), round(-l, 6)}
        self.size_ll = {}
        V = P["sizes"][self.ty]
        need = {(pi, li) for (pi, s, li, z) in self.hyp}
        for li, lag in enumerate(P["size_lags"]):
            for z in (1, -1):
                L = 0.0
                for t in self.ts:
                    v = self.s[t]
                    if v == 0:
                        continue
                    b = self.zbit(t, lag)
                    k = PAIR_INDEX[v] - 1
                    if b is None:
                        L += math.log(.5 * (V["hi"][k] + V["lo"][k]))
                    else:
                        L += math.log(V["hi" if b * z > 0 else "lo"][k])
                self.size_ll[(li, z)] = L
        self.pre = {}
        for pi, pair in enumerate(P["pairs"]):
            for li, lag in enumerate(P["size_lags"]):
                if (pi, li) not in need:
                    continue
                for z in (1, -1):
                    arr = {v: [0.0] for v in vals}
                    for t in self.ts:
                        won = self.s[t] > 0
                        x = self.bits(t, pair)
                        b = self.zbit(t, lag)
                        zv = None if b is None else b * z
                        o = offsets.get(t, 0.0)
                        for v, a in arr.items():
                            a.append(a[-1] + self.tick_ll(t, won, x, zv, v, o))
                    self.pre[(pi, li, z)] = arr
        n = len(self.ts)
        self.stable, self.flip, self.flipc = {}, {}, {}
        tie = P.get("lag_tie", 0.0)
        for h in self.hyp:
            pi, s, li, z = h
            A = self.pre[(pi, li, z)]
            extra = self.size_ll[(li, z)]
            if 0 < tie < 1:
                # the size lag usually equals the first lag of the win pair
                same = P["size_lags"][li] == P["pairs"][pi][0]
                extra += math.log(tie if same else (1 - tie) / (len(P["size_lags"]) - 1))
            self.stable[h] = A[float(s)][n] + extra
            fm = self._flip_ll(A, float(s), n, w_m, J)
            self.flip[h] = [v + extra for v in fm]
            if w_c == w_m:
                self.flipc[h] = self.flip[h]
            else:
                self.flipc[h] = [v + extra for v in self._flip_ll(A, float(s), n, w_c, J)]
        self.mult_m = [mult_at(self.cut, tau, w_m, J) for tau in self.taus]
        self.mult_c = [mult_at(self.cut, tau, w_c, J) for tau in self.taus]

    def eta_at(self, h, t, mult, H):
        """Return (x, z, S) at tick t; x/z may be unknown -> None handled by caller."""
        P = self.P
        pi, s, li, z = h
        x = self.bits(t, P["pairs"][pi])
        b = self.zbit(t, P["size_lags"][li])
        return x, (None if b is None else b * z), s * mult

    def vector(self, h, mult, Hprob):
        """Five-payout probabilities at the cutoff under a hypothesis and rule multiplier."""
        P = self.P
        x, z, S = self.eta_at(h, self.cut, mult, 0)
        V = P["sizes"][self.ty]
        out = [0.0] * 5
        zs = ((z, 1.0),) if z is not None else ((1, .5), (-1, .5))
        xs = ((x, 1.0),) if x is not None else ((1, .5), (-1, .5))
        d = P["delta"]
        b0 = self.base() + self.xo.get(self.cut, 0.0)
        for zv, zw in zs:
            for xv, xw in xs:
                for H, hw in ((1, Hprob), (-1, 1 - Hprob)):
                    w = sig(b0 + P["K"] * S * xv + P["gamma"][self.ty] * zv + d * H)
                    m = zw * xw * hw
                    out[0] += m * (1 - w)
                    for k in range(4):
                        out[k + 1] += m * w * V["hi" if zv > 0 else "lo"][k]
        return out


def is_short(machines):
    """An episode is short-history when its machines saw far fewer ticks than the cutoff."""
    P = machines[0].P
    return max(len(m.ts) for m in machines) <= P.get("short_len", 100)


def condition_posteriors(machines):
    P = machines[0].P
    Es = 0.0
    nt = len(P["taus"])
    Ed = 0.0
    for m in machines:
        lpH = -math.log(len(m.hyp))
        Es += lse(list(m.stable.values())) + lpH
        Ed += lse([lse(m.flip[h]) for h in m.hyp]) - math.log(nt) + lpH
    per_tau = [sum(lse([m.flipc[h][i] for h in m.hyp]) - math.log(len(m.hyp)) for m in machines)
               for i in range(nt)]
    Ec = lse(per_tau) - math.log(nt)
    pri = P["cond_prior"]
    if "cond_prior_short" in P and is_short(machines):
        # sparse (short) histories: train shows these episodes behave like stable ones
        pri = P["cond_prior_short"]
    lw = [Es + math.log(pri[0]), Ed + math.log(pri[1]), Ec + math.log(pri[2])]
    Z = lse(lw)
    return [math.exp(v - Z) for v in lw], per_tau


def machine_weights(m, W, per_tau):
    """Return dict (h, rule multiplier at cutoff) -> weight, marginal over condition and tau."""
    nt = len(m.P["taus"])
    out = defaultdict(float)
    # no flip
    Z = lse(list(m.stable.values()))
    for h, v in m.stable.items():
        out[(h, 1.0)] += W[0] * math.exp(v - Z)
    # per-machine flip
    if W[1] > 1e-9:
        Z = lse([lse(m.flip[h]) for h in m.hyp])
        for h in m.hyp:
            for i, v in enumerate(m.flip[h]):
                e = math.exp(v - Z)
                if e > 1e-9:
                    out[(h, m.mult_m[i])] += W[1] * e
    # common flip, leave this machine's marginal out and put it back per hypothesis
    if W[2] > 1e-9:
        lhm = math.log(len(m.hyp))
        lp = [per_tau[i] - (lse([m.flipc[h][i] for h in m.hyp]) - lhm) for i in range(nt)]
        cell = {h: [lp[i] + m.flipc[h][i] for i in range(nt)] for h in m.hyp}
        Z = lse([v for vals in cell.values() for v in vals])
        for h, vals in cell.items():
            for i, v in enumerate(vals):
                e = math.exp(v - Z)
                if e > 1e-9:
                    out[(h, m.mult_c[i])] += W[2] * e
    return out


def best_path(m, W, per_tau):
    """MAP hypothesis used to extract the episode hot/cold signal: (h, tau, width)."""
    w_m, w_c, J = widths(m.P)
    if W[0] >= max(W[1], W[2]):
        h = max(m.stable, key=m.stable.get)
        return h, None, 0
    if W[2] >= W[1]:
        i = max(range(len(per_tau)), key=per_tau.__getitem__)
        h = max(m.hyp, key=lambda k: m.flipc[k][i])
        return h, m.P["taus"][i], w_c
    best, tau = None, None
    for h in m.hyp:
        for i, v in enumerate(m.flip[h]):
            if best is None or v > best[0]:
                best, tau = (v, h), m.P["taus"][i]
    return best[1], tau, w_m


def episode_hot(machines, W, per_tau):
    """Forward-backward over H(t) given each machine's MAP rule. Returns P(H=+1) at cutoff, offsets."""
    P = machines[0].P
    J = widths(P)[2]
    cut = machines[0].cut
    paths = [best_path(m, W, per_tau) for m in machines]
    ev = defaultdict(lambda: [0.0, 0.0])
    for m, (h, tau, wd) in zip(machines, paths):
        pi, s, li, z = h
        for t in m.ts:
            x = m.bits(t, P["pairs"][pi])
            b = m.zbit(t, P["size_lags"][li])
            zv = None if b is None else b * z
            S = s * mult_at(t, tau, wd, J)
            won = m.s[t] > 0
            ev[t][0] += m.tick_ll(t, won, x, zv, S, P["delta"])
            ev[t][1] += m.tick_ll(t, won, x, zv, S, -P["delta"])
    T = sorted(ev)
    rho = P["rho"]
    f = []
    a = [.5, .5]
    for i, t in enumerate(T):
        gap = t - T[i - 1] if i else 1
        r = .5 * (1 - (1 - 2 * rho) ** gap)
        a = [a[0] * (1 - r) + a[1] * r, a[1] * (1 - r) + a[0] * r]
        m0 = max(ev[t])
        a = [a[0] * math.exp(ev[t][0] - m0), a[1] * math.exp(ev[t][1] - m0)]
        z = sum(a)
        a = [v / z for v in a]
        f.append(a)
    gap = cut - T[-1] if T else 1
    r = .5 * (1 - (1 - 2 * rho) ** gap)
    pred = a[0] * (1 - r) + a[1] * r
    b = [1.0, 1.0]
    off = {}
    for i in range(len(T) - 1, -1, -1):
        t = T[i]
        g = [f[i][0] * b[0], f[i][1] * b[1]]
        z = sum(g)
        off[t] = P["delta"] * (g[0] - g[1]) / z
        m0 = max(ev[t])
        e = [math.exp(ev[t][0] - m0), math.exp(ev[t][1] - m0)]
        gapb = t - T[i - 1] if i else 1
        r = .5 * (1 - (1 - 2 * rho) ** gapb)
        nb = [(1 - r) * e[0] * b[0] + r * e[1] * b[1], r * e[0] * b[0] + (1 - r) * e[1] * b[1]]
        z = max(nb)
        b = [v / z for v in nb]
    return pred, off


def machine_effects(machines, W, per_tau, offsets):
    """One Newton step of a shrunken per-machine logit offset along the MAP rule."""
    P = machines[0].P
    J = widths(P)[2]
    tau2 = P.get("re_var", 0.0)
    out = {}
    if tau2 <= 0:
        return out
    for m in machines:
        h, tau, wd = best_path(m, W, per_tau)
        pi, s, li, z = h
        g, I = 0.0, 1.0 / tau2
        for t in m.ts:
            x = m.bits(t, P["pairs"][pi])
            b = m.zbit(t, P["size_lags"][li])
            if x is None or b is None:
                continue
            S = s * mult_at(t, tau, wd, J)
            pw = sig(m.base() + m.xo.get(t, 0.0) + offsets.get(t, 0.0) + P["K"] * S * x + P["gamma"][m.ty] * b * z)
            g += (1.0 if m.s[t] > 0 else 0.0) - pw
            I += pw * (1 - pw)
        out[machine_key(m.q)] = g / I
    return out


def cross_counts(queries, history, lag):
    """Per machine and tick: how many OTHER machines in the episode won `lag` ticks earlier.
    Unobserved peers are filled in at the observed peers' rate (None when no peer is observed)."""
    keys = [machine_key(q) for q in queries]
    out = {}
    for q, k in zip(queries, keys):
        ticks = [t for t in history[k] if t < q["cutoff"]] + [q["cutoff"]]
        peers = [history[j] for j in keys if j != k]
        d = {}
        for t in ticks:
            obs = [p[t - lag] > 0 for p in peers if (t - lag) in p]
            d[t] = None if not obs else sum(obs) * len(peers) / len(obs)
        out[k] = d
    return out


def cross_offsets(queries, history, P):
    """Known logit offset from other machines' earlier wins: lam * (count - c0)."""
    lag = P.get("cross_lag", 0)
    if not lag:
        return {}
    lam, c0 = P["cross_lam"], P["cross_c0"]
    out = {}
    for k, d in cross_counts(queries, history, lag).items():
        out[k] = {t: lam * (c - c0) for t, c in d.items() if c is not None}
    return out


def predict_episode(queries, history, P):
    Hprob = .5
    offsets = {}
    effects = {}
    xo = cross_offsets(queries, history, P)
    iters = P["iters"]
    for it in range(iters):
        machines = [Machine(q, history[machine_key(q)], P, offsets, effects, xo) for q in queries]
        W, per_tau = condition_posteriors(machines)
        if it == iters - 1 and it > 0 and P.get("re_var", 0) > 0 and not effects:
            effects = machine_effects(machines, W, per_tau, offsets)
            machines = [Machine(q, history[machine_key(q)], P, offsets, effects, xo) for q in queries]
            W, per_tau = condition_posteriors(machines)
        if it < iters - 1 and P["delta"] > 0:
            Hprob, offsets = episode_hot(machines, W, per_tau)
    if P["delta"] > 0 and P["iters"] == 1:
        Hprob, _ = episode_hot(machines, W, per_tau)
    out = {}
    for m in machines:
        vec = [0.0] * 5
        for (h, mult), w in machine_weights(m, W, per_tau).items():
            if w < 1e-6:
                continue
            v = m.vector(h, mult, Hprob)
            for k in range(5):
                vec[k] += w * v[k]
        z = sum(vec)
        out[m.q["query_id"]] = [v / z for v in vec]
    return out, W, Hprob


def predict_all(episodes, history, P):
    preds, info = {}, {}
    for e in sorted(episodes):
        pr, W, Hp = predict_episode(episodes[e], history, P)
        preds.update(pr)
        info[e] = (W, Hp)
    return preds, info


def load_params(path):
    P = json.loads(Path(path).read_text(encoding="utf-8"))
    P["pairs"] = [tuple(p) for p in P["pairs"]]
    return P

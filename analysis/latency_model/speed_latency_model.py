#!/usr/bin/env python3
"""Speed-latency model for VLALagBench success curves.

1. Transfer model: every (mode, latency) line is the synchronous-0 ms curve at a
   rescaled speed,
       S_mode,L(v) = S0(kappa * v),
       kappa = 1 + a * dstale + b * pause,
   where dstale (control steps) is the extra mean age of the observation behind each
   executed action and pause = L/H is the idle time per unit of acting time
   (synchronous only). Both follow exactly from vlalagbench.timing.ChunkScheduler. a and
   b are two per-task sensitivities. kappa is the paper's equivalent speedup lambda.

   Two optional terms, tested blind in refined.py (the paper uses the plain law, M1):
   - plan-tail cost c: success is multiplied by rho = exp(-c * max(0, i - 4)), where i
     is the mean chunk index of the executed actions; long-latency lazy and continuous
     lines execute the far end of each plan. One constant per policy (pi0.5: 0.026).
   - interception gamma: above about 2 v50 a hand that holds still can be reached by
     the target coming round its path, so a failure survives each synchronous pause
     with probability exp(-gamma * v * L * dt).

2. Reference curve (Reference): S0, or the line S_X a calibration starts from, is the
   measured curve itself, linear between measured speeds and not forced to fall with
   speed. Only past the fastest measured speed does it follow the log-logistic
       S0(v) = s * (f + (1 - f) / (1 + (v / v50) ** beta))
   fitted to the same points, scaled to meet the measured rate there. Every prediction
   of the paper reads this curve (predict_from, transfer, fit_ab); predict(), probe()
   and the fit and probe commands keep the whole-curve log-logistic for comparison.

3. Adaptive estimation of S0 for a new task: grid posterior over (s, v50, beta, f)
   with a prior from the pi0.5 tasks measured so far; each batch goes to the speed
   that most reduces the posterior variance of the curve over the speeds you report.

4. Two probes instead of a latency grid: one lazy and one synchronous point at the
   same latency, near v50 / 1.4, each inverted through S0 to the speed it acts like.
   The lazy probe gives a; the synchronous probe's excess over it gives b.

5. Transfer from one complete line at a nonzero delay: S_Y(v) = S_X(v * kappa_Y / kappa_X),
   so one complete line X of any mode, as its measured Reference, carries the curve's shape and
   one point on each of two other lines fixes (a, b); the never-measured sync-0 curve follows.
   Both points run at the grid speed nearest the reference's midpoint speed / 1.4, where it
   first falls halfway between its highest and lowest measured success.

Which lines identify (a, b), in every strategy: a synchronous line at L steps has
kappa = 1 + (a + b/H) L, so synchronous lines alone fix only a + b/H; an asynchronous
line has no pause, so it fixes a and never b; sync 0 has kappa = 1 and fixes neither.
The data must therefore hold at least one synchronous and one asynchronous line at a
nonzero delay (separates_age_and_pause). Nothing else is required.

Commands
  fit       CSV with columns line,v,k,n (line = sync0, sync100, lazy200, cont100, ...)
            prints S0, a, b (and c, gamma when asked), every line's kappa and the fit error.
  next      CSV of sync-0 episodes so far (v,k,n): prints the current curve with 90%
            bands and the speed for the next batch.
  predict   prints the predicted curve of any mode/latency from S0 parameters and
            (a, b), optionally with c and gamma.
  probe     CSV of sync-0 episodes (v,k,n): prints the probe speed; given one lazy and
            one synchronous probe point, a and b with their sampling spread.
  transfer  CSV of one complete line (v,k,n) and its setting (--line lazy100): prints the
            probe speed; given one point on each of two other lines (--probe sync200,v,k,n),
            a and b with their spread, every setting's kappa, and the zero-delay curve.
  validate  blind test on the four LIBERO tasks (fit on 0/100/200 ms, predict 50/150/250 ms), for pi0.5 or
            --policy MolmoAct2; reads the published result tables (refined.py).

fit, next, predict, probe and transfer need only numpy and scipy.
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np
from scipy import optimize, special, stats


HZ = 20          # control rate (Hz)
H = 5            # execution horizon (actions executed per chunk)
TAIL_START = 4   # the plan-tail cost starts past this mean executed chunk index (pi0.5, H = 5)
C_PI05 = 0.026   # pi0.5 plan-tail constant, fitted jointly on all four LIBERO tasks (refined.py, M4)
PROBE_FACTOR = 1.4   # probes run at v50 / PROBE_FACTOR

# Prior over the sync-0 curve from six pi0.5 LIBERO task families
# (basket, plate, bowl on cabinet, drawer, cream cheese, salad dressing).
POP_PRIOR = {
    "logit_s": (2.86, 0.94),      # s median 0.946
    "log_v50": (-2.61, 0.70),     # v50 median 0.073 m/s, x/÷ 2.0 per SD
    "log_beta": (1.37, 0.35),     # beta median 3.9
    "logit_f": (-1.99, 1.77),     # floor median 0.12
}


# ----------------------------------------------------------------------- curves
def s0_curve(v, s, v50, beta, f=0.0):
    v = np.maximum(np.asarray(v, float), 1e-9)
    # 1 / (1 + (v/v50)^beta), written to avoid overflow
    return s * (f + (1.0 - f) * special.expit(-beta * (np.log(v) - np.log(v50))))


def latency_steps(latency_ms: float, hz: int = HZ) -> int:
    """Latency in control steps, rounded up as vlalagbench.timing.effective_steps does."""
    return int(math.ceil(latency_ms * hz / 1000.0 - 1e-9))


def _mode(mode: str) -> str:
    mode = mode.lower()[:4]
    if mode not in ("sync", "lazy", "cont"):
        raise ValueError(f"unknown mode {mode!r}")
    return mode


def schedule(mode: str, L: int, horizon: int = H) -> dict:
    """Steady-state executed actions of vlalagbench.timing.ChunkScheduler at L latency steps.

    Synchronous waits L idle steps, then executes chunk[0..H-1]; lazy requests early and
    executes chunk[L..L+H-1]; continuous replaces the plan every L steps and executes
    chunk[L..2L-1]. An executed action at chunk index i acts on an observation i steps
    old (synchronous: L + i). Returns the mean observation age (steps), the mean chunk
    index, idle steps per acting step, and the number of actions executed per chunk.
    """
    mode = _mode(mode)
    if mode == "sync":
        idx = np.arange(horizon)
        age = L + idx
        pause = L / horizon
    elif mode == "lazy":
        idx = np.arange(L, L + horizon)
        age = idx
        pause = 0.0
    else:
        if L > horizon:
            raise ValueError("continuous execution with L > H idles between chunks; not modelled")
        idx = np.arange(L, 2 * L) if L > 0 else np.array([0])
        age = idx
        pause = 0.0
    return dict(age=float(np.mean(age)), idx=float(np.mean(idx)), pause=float(pause), span=len(idx))


def schedule_features(mode: str, latency_ms: float, horizon: int = H, hz: int = HZ, ref_horizon: int = None):
    """(dstale, pause) for one inference setting, relative to synchronous 0 ms.

    dstale: extra mean observation age of executed actions, in control steps
            (synchronous and lazy: L + (horizon - 1)/2 - (ref_horizon - 1)/2; continuous: (3L - 1)/2 - (ref_horizon - 1)/2).
    pause:  idle steps per acting step (synchronous only: L/horizon).
    ref_horizon is the horizon of the synchronous 0 ms schedule that defines kappa = 1 (default: horizon). A schedule
    run at another horizon than the reference curve's, such as lazy execution with a shorter horizon, passes the
    reference's horizon here.
    """
    s = schedule(mode, latency_steps(latency_ms, hz), horizon)
    return s["age"] - schedule("sync", 0, ref_horizon or horizon)["age"], s["pause"]


def kappa(mode: str, latency_ms: float, a: float, b: float, horizon: int = H, hz: int = HZ) -> float:
    ds, pause = schedule_features(mode, latency_ms, horizon, hz)
    return max(1.0 + a * ds + b * pause, 1e-3)


def plan_tail(mode: str, latency_ms: float, c: float, horizon: int = H, hz: int = HZ,
              tail_start: float = TAIL_START) -> float:
    """Success factor rho = exp(-c * max(0, i - tail_start)) for executing late chunk actions."""
    if not c:
        return 1.0
    i = schedule(mode, latency_steps(latency_ms, hz), horizon)["idx"]
    return math.exp(-c * max(0.0, i - tail_start))


def predict(v, mode, latency_ms, s, v50, beta, f, a, b, horizon=H, hz=HZ, c=0.0, gamma=0.0):
    v = np.asarray(v, float)
    base = s0_curve(v * kappa(mode, latency_ms, a, b, horizon, hz), 1.0, v50, beta, f)
    if gamma and _mode(mode) == "sync":
        base = 1 - (1 - base) * np.exp(-gamma * v * latency_steps(latency_ms, hz) / hz)
    return plan_tail(mode, latency_ms, c, horizon, hz) * s * base


def parse_line(line: str):
    line = line.strip().lower()
    for m in ("sync", "lazy", "cont"):
        if line.startswith(m):
            return m, float(line[len(m):] or 0)
    raise ValueError(f"cannot parse line name {line!r} (expected e.g. sync0, lazy100, cont200)")


# ----------------------------------------------------------------------- fitting
def _ll(k, n, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return k * np.log(p) + (n - k) * np.log1p(-p)


def fit(rows, horizon=H, hz=HZ, fit_ab=True, prior=POP_PRIOR, c=0.0, fit_c=False, fit_gamma=False):
    """rows: list of (line, v, k, n). Returns dict of parameters.

    prior: Gaussian prior on (logit s, log v50, log beta, logit f), plus a weak ridge on
    (a, b) with sd 0.5 and 3. It only matters when data are sparse; pass prior=None for
    plain maximum likelihood.
    c: fixed plan-tail constant (C_PI05 for pi0.5), or fitted with fit_c.
    fit_gamma: also fit the interception term of the synchronous lines.
    """
    lines = [parse_line(r[0]) for r in rows]
    v = np.array([float(r[1]) for r in rows])
    k = np.array([float(r[2]) for r in rows])
    n = np.array([float(r[3]) for r in rows])
    feats = np.array([schedule_features(m, L, horizon, hz) for m, L in lines])
    tail = np.array([max(0.0, schedule(m, latency_steps(L, hz), horizon)["idx"] - TAIL_START) for m, L in lines])
    lsync = np.array([latency_steps(L, hz) if m == "sync" else 0 for m, L in lines], float)
    fit_c, fit_gamma = fit_c and fit_ab, fit_gamma and fit_ab

    bounds = [(-6.0, 9.0), (math.log(1e-3), math.log(5.0)), (math.log(0.2), math.log(30.0)), (-9.0, 3.0)]
    if fit_ab:
        bounds += [(-1.0, 3.0), (-3.0, 10.0)]
    bounds += [(0.0, 0.5)] * fit_c + [(0.0, 200.0)] * fit_gamma

    def unpack(x):
        s, v50, beta, f = special.expit(x[0]), float(np.exp(x[1])), float(np.exp(x[2])), special.expit(x[3])
        a, b = (x[4], x[5]) if fit_ab else (0.0, 0.0)
        extra = list(x[6:] if fit_ab else [])
        cc = extra.pop(0) if fit_c else c
        gamma = extra.pop(0) if fit_gamma else 0.0
        return s, v50, beta, f, a, b, cc, gamma

    def model(x):
        s, v50, beta, f, a, b, cc, gamma = unpack(x)
        kap = np.maximum(1 + a * feats[:, 0] + b * feats[:, 1], 1e-3)
        base = s0_curve(v * kap, 1.0, v50, beta, f)
        if gamma:
            base = 1 - (1 - base) * np.exp(-gamma * v * lsync / hz)
        return np.exp(-cc * tail) * s * base

    def penalty(x):
        if prior is None:
            return 0.0
        lf = math.log(max(special.expit(x[3]), 0.01) / max(1 - special.expit(x[3]), 1e-9))
        terms = [((x[0] - prior["logit_s"][0]) / prior["logit_s"][1]) ** 2,
                 ((x[1] - prior["log_v50"][0]) / prior["log_v50"][1]) ** 2,
                 ((x[2] - prior["log_beta"][0]) / prior["log_beta"][1]) ** 2,
                 ((lf - prior["logit_f"][0]) / prior["logit_f"][1]) ** 2]
        if fit_ab:
            terms += [(x[4] / 0.5) ** 2, (x[5] / 3.0) ** 2]
        return 0.5 * sum(terms)

    def nll(x):
        return -_ll(k, n, model(x)).sum() + penalty(x)

    best = None
    for lv in (np.log(0.03), np.log(0.08), np.log(0.2)):
        for lb in (np.log(2.0), np.log(4.0)):
            for ab in ([0.05, 0.3], [0.15, 0.0], [0.0, 1.5]):
                for extra in ([0.02] * fit_c + [g] * fit_gamma for g in ((0.0, 5.0) if fit_gamma else (0.0,))):
                    x0 = [3.0, lv, lb, -1.5] + (ab + extra if fit_ab else [])
                    r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
                    if best is None or r.fun < best.fun:
                        best = r
    r = optimize.minimize(nll, best.x, method="Powell", bounds=bounds,
                          options=dict(maxiter=20000, xtol=1e-7, ftol=1e-11))
    x = r.x if r.fun < best.fun else best.x
    s, v50, beta, f, a, b, cc, gamma = unpack(x)
    p = model(x)
    return dict(s=s, v50=v50, beta=beta, f=f, a=a, b=b, c=cc, gamma=gamma, pred=p,
                rmse_pp=float(np.sqrt(np.mean((k / n - p) ** 2)) * 100))


# ----------------------------------------------------------------------- probes
def invert_s0(p, s, v50, beta, f, margin=0.01):
    """The speed at which S0 succeeds with probability p, kept inside the curve's range."""
    p = float(np.clip(p, s * f + margin, s - margin))
    return optimize.brentq(lambda x: s0_curve(x, s, v50, beta, f) - p, 1e-4, 10.0)


def probe(s0, lazy, sync, latency_ms, horizon=H, hz=HZ, draws=300, seed=0):
    """a and b from one lazy and one synchronous probe at the same latency.

    s0: (s, v50, beta, f) of the task's sync-0 curve; lazy, sync: (v, k, n) of each probe.
    Each probe's success rate is inverted through S0 to the speed it acts like, and
    kappa = that speed / the probe speed. Lazy has the synchronous observation age but no
    pause, so a = (kappa_lazy - 1) / dstale and b = (kappa_sync - kappa_lazy) / pause.
    The spread redraws both probe counts binomially.
    """
    dstale = schedule_features("lazy", latency_ms, horizon, hz)[0]
    pause = schedule_features("sync", latency_ms, horizon, hz)[1]

    def kap(point, k=None):
        v, kk, n = point
        return invert_s0((kk if k is None else k) / n, *s0) / v

    def ab(kl, ks):
        return (kl - 1) / dstale, (ks - kl) / pause

    a, b = ab(kap(lazy), kap(sync))
    rng = np.random.default_rng(seed)
    spread = np.array([ab(kap(lazy, rng.binomial(lazy[2], lazy[1] / lazy[2])),
                          kap(sync, rng.binomial(sync[2], sync[1] / sync[2]))) for _ in range(draws)])
    return dict(a=float(a), b=float(b), a_sd=float(spread[:, 0].std()), b_sd=float(spread[:, 1].std()),
                kappa_lazy=kap(lazy), kappa_sync=kap(sync))


# ----------------------------------------------------------------------- transfer
def separates_age_and_pause(lines) -> bool:
    """Whether these lines identify (a, b): at least one synchronous and one asynchronous line at a nonzero delay.

    Synchronous lines move kappa by (a + b/H) L, asynchronous lines by a * dstale; sync 0 does not move it.
    """
    settings = [parse_line(line) for line in lines]
    return (any(m == "sync" and lat > 0 for m, lat in settings)
            and any(m != "sync" and lat > 0 for m, lat in settings))


TRANSFER_A = np.arange(-0.30, 0.80 + 1e-9, 0.005)   # grid over a (per control step of observation age)
TRANSFER_B = np.arange(-1.00, 4.00 + 1e-9, 0.025)   # grid over b (per unit pause ratio)
TRANSFER_PRIOR_SD = (0.5, 2.0)                       # weak prior on (a, b); decides only when probes carry no information


def transfer_grid():
    """Flattened (a, b) grid and its log prior."""
    a, b = (g.ravel() for g in np.meshgrid(TRANSFER_A, TRANSFER_B, indexing="ij"))
    return a, b, -0.5 * ((a / TRANSFER_PRIOR_SD[0]) ** 2 + (b / TRANSFER_PRIOR_SD[1]) ** 2)


def transfer_curves(reference, ref_feat, feats, v, a, b):
    """Success of points on other lines from the reference line's measured curve, for every (a, b) on the grid.

    S_Y(v) = S0(kappa_Y v) = S_X(v kappa_Y / kappa_X): the reference line X carries the shape, (a, b) the
    ratio. reference: the Reference of S_X; ref_feat, feats: (dstale, pause) of X and of each point's line.
    Returns (grid, points) probabilities and the mask where both kappas are positive.
    """
    feats = np.asarray(feats, float).reshape(-1, 2)
    kx = 1 + a * ref_feat[0] + b * ref_feat[1]
    ky = 1 + a[:, None] * feats[None, :, 0] + b[:, None] * feats[None, :, 1]
    ok = (kx[:, None] > 0.05) & (ky > 0.05)
    ratio = np.where(ok, ky / np.maximum(kx[:, None], 0.05), 1.0)
    return reference(np.asarray(v, float)[None, :] * ratio), ok


def transfer(ref_points, ref_line, probes, horizon=H, hz=HZ, draws=300, seed=0):
    """(a, b) from one complete line and single points on other lines.

    ref_points: (v, k, n) of the complete line ref_line (any mode and latency, e.g. sync0 or lazy100);
    probes: (line, v, k, n), one point per line. The reference is the line's measured curve (Reference);
    (a, b) maximise the probes' likelihood under S_X(v kappa_Y / kappa_X) on the grid. Two probes fix both
    sensitivities when the three lines separate age from pause (at least one synchronous and one asynchronous
    line at a nonzero delay). The spread redraws every probe count binomially. Returns the reference, a, b,
    their spread, and the reference line's own kappa.
    """
    if len(probes) >= 2 and not separates_age_and_pause([ref_line] + [p[0] for p in probes]):
        raise ValueError("these lines cannot separate a from b: include at least one synchronous and one "
                         "asynchronous line at a nonzero delay")
    m, lat = parse_line(ref_line)
    reference = Reference(ref_points)
    a, b, log_prior = transfer_grid()
    ref_feat = schedule_features(m, lat, horizon, hz)
    feats = [schedule_features(*parse_line(p[0]), horizon, hz) for p in probes]
    v = np.array([float(p[1]) for p in probes])
    k = np.array([float(p[2]) for p in probes])
    n = np.array([float(p[3]) for p in probes])
    p, ok = transfer_curves(reference, ref_feat, feats, v, a, b)

    def best(kk):
        return int(np.argmax(np.where(ok, _ll(kk[None, :], n[None, :], p), -1e9).sum(1) + log_prior))

    g = best(k)
    rng = np.random.default_rng(seed)
    spread = np.array([[a[j], b[j]] for j in (best(rng.binomial(n.astype(int), k / n).astype(float))
                                              for _ in range(draws))]) if draws else np.zeros((1, 2))
    kx = 1 + a[g] * ref_feat[0] + b[g] * ref_feat[1]
    return dict(reference=reference, ref_feat=tuple(ref_feat), a=float(a[g]), b=float(b[g]),
                a_sd=float(spread[:, 0].std()), b_sd=float(spread[:, 1].std()), kappa_ref=float(kx))


# ----------------------------------------------------------------------- measured reference curve
class Reference:
    """One complete measured success curve, the curve every prediction reads.

    Between measured speeds it is linear in the measured success rates, and below the slowest speed it keeps that
    speed's rate; nothing forces it to fall with speed. Past the fastest measured speed, where nothing was measured,
    it follows the log-logistic S0 fitted to the same points, multiplied by the constant that makes it meet the
    measured rate at the fastest speed. The fitted curve is used nowhere else.
    points: (v, k, n) of one line; repeated speeds are pooled.
    """

    def __init__(self, points):
        pooled = {}
        for v, k, n in points:
            kk, nn = pooled.get(float(v), (0.0, 0.0))
            pooled[float(v)] = (kk + float(k), nn + float(n))
        self.v = np.array(sorted(pooled))
        self.k = np.array([pooled[v][0] for v in self.v])
        self.n = np.array([pooled[v][1] for v in self.v])
        self.p = self.k / self.n
        res = fit([("sync0", v, k, n) for v, k, n in zip(self.v, self.k, self.n)], fit_ab=False, prior=None)
        self.tail = (res["s"], res["v50"], res["beta"], res["f"])
        self.edge = float(s0_curve(self.v[-1], *self.tail))

    def __call__(self, u):
        u = np.asarray(u, float)
        inside = np.interp(u, self.v, self.p)
        if self.edge <= 1e-12:
            return np.where(u > self.v[-1], self.p[-1], inside)
        return np.where(u > self.v[-1], self.p[-1] * s0_curve(u, *self.tail) / self.edge, inside)

    def midpoint_speed(self):
        """The speed where the measured curve first falls through halfway between its highest and lowest rate,
        after its highest; the fastest measured speed if it never does."""
        half = (self.p.max() + self.p.min()) / 2
        for i in range(int(np.argmax(self.p)), len(self.v) - 1):
            if self.p[i] >= half > self.p[i + 1]:
                return float(self.v[i] + (self.p[i] - half) / (self.p[i] - self.p[i + 1]) * (self.v[i + 1] - self.v[i]))
        return float(self.v[-1])

    def probe_speed(self, speeds, factor=1 / PROBE_FACTOR):
        """The grid speed nearest factor times the midpoint speed, where both probes run."""
        speeds = np.asarray(speeds, float)
        return float(speeds[np.argmin(np.abs(speeds - factor * self.midpoint_speed()))])


AB_BOUNDS = ((-1.0, 3.0), (-3.0, 10.0))   # a per control step of observation age, b per unit pause ratio
AB_GRID = (np.arange(-1.0, 3.0 + 1e-9, 0.01), np.arange(-3.0, 10.0 + 1e-9, 0.05))


def speed_ratio(feats, ref_feat, a, b):
    """kappa_Y / kappa_X of each point's setting against the reference line's, kappas floored as kappa() does."""
    feats = np.asarray(feats, float).reshape(-1, 2)
    kx = max(1 + a * ref_feat[0] + b * ref_feat[1], 1e-3)
    return np.maximum(1 + a * feats[:, 0] + b * feats[:, 1], 1e-3) / kx


def fit_ab(reference, ref_feat, feats, v, k, n):
    """Maximum-likelihood (a, b) of points on other lines read from the reference at v kappa_Y / kappa_X.

    The likelihood is not smooth (the reference is piecewise linear), so the grid AB_GRID picks the start and
    Nelder-Mead refines it inside AB_BOUNDS.
    """
    feats = np.asarray(feats, float).reshape(-1, 2)
    v, k, n = (np.asarray(x, float) for x in (v, k, n))
    ga, gb = AB_GRID
    best = (-np.inf, 0.0, 0.0)
    for a in ga:
        kx = np.maximum(1 + a * ref_feat[0] + gb * ref_feat[1], 1e-3)
        ky = np.maximum(1 + a * feats[None, :, 0] + gb[:, None] * feats[None, :, 1], 1e-3)
        ll = _ll(k[None, :], n[None, :], reference(v[None, :] * ky / kx[:, None])).sum(1)
        j = int(np.argmax(ll))
        if ll[j] > best[0]:
            best = (float(ll[j]), float(a), float(gb[j]))
    (alo, ahi), (blo, bhi) = AB_BOUNDS

    def nll(x):
        a, b = float(np.clip(x[0], alo, ahi)), float(np.clip(x[1], blo, bhi))
        return -_ll(k, n, reference(v * speed_ratio(feats, ref_feat, a, b))).sum()

    r = optimize.minimize(nll, [best[1], best[2]], method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-9, maxiter=4000, initial_simplex=[
                              [best[1], best[2]], [best[1] + 0.01, best[2]], [best[1], best[2] + 0.05]]))
    if r.fun < -best[0]:
        return float(np.clip(r.x[0], alo, ahi)), float(np.clip(r.x[1], blo, bhi))
    return best[1], best[2]


def predict_from(reference, v, mode, latency_ms, a, b, horizon=H, hz=HZ, ref_feat=(0.0, 0.0), ref_horizon=None):
    """Success of one setting read from the reference: S_X(v kappa / kappa_ref). ref_feat is the reference line's
    (dstale, pause), zero for synchronous 0 ms; ref_horizon as in schedule_features."""
    feat = schedule_features(mode, latency_ms, horizon, hz, ref_horizon)
    return reference(np.atleast_1d(np.asarray(v, float)) * speed_ratio([feat], ref_feat, a, b)[0])


# ----------------------------------------------------------------------- adaptive S0
class AdaptiveS0:
    """Grid posterior over (s, v50, beta, f) for a task's sync-0 curve."""

    def __init__(self, prior=POP_PRIOR):
        ls = np.linspace(0.0, 5.3, 11)
        lv = np.linspace(np.log(0.008), np.log(1.0), 50)
        lb = np.linspace(np.log(1.0), np.log(10.0), 13)
        ff = np.array([0.0, 0.03, 0.08, 0.15, 0.25, 0.35, 0.5])
        A, B, C, D = np.meshgrid(ls, lv, lb, ff, indexing="ij")
        self.s, self.v50, self.beta, self.f = special.expit(A).ravel(), np.exp(B).ravel(), np.exp(C).ravel(), D.ravel()
        lf = special.logit(np.clip(self.f, 0.01, 0.99))
        self.logw = (stats.norm.logpdf(A.ravel(), *prior["logit_s"]) + stats.norm.logpdf(B.ravel(), *prior["log_v50"])
                     + stats.norm.logpdf(C.ravel(), *prior["log_beta"]) + stats.norm.logpdf(lf, *prior["logit_f"]))
        self._norm()

    def _norm(self):
        self.logw -= self.logw.max()
        w = np.exp(self.logw)
        self.w = w / w.sum()

    def _p(self, v, idx):
        return s0_curve(v, self.s[idx], self.v50[idx], self.beta[idx], self.f[idx])

    def update(self, v, k, n):
        p = np.clip(s0_curve(v, self.s, self.v50, self.beta, self.f), 1e-9, 1 - 1e-9)
        self.logw += k * np.log(p) + (n - k) * np.log1p(-p)
        self._norm()

    def _active(self, tol=1e-7):
        idx = np.where(self.w > tol * self.w.max())[0]
        return idx, self.w[idx] / self.w[idx].sum()

    def band(self, vs, q=(0.05, 0.5, 0.95)):
        idx, w = self._active(1e-9)
        out = []
        for v in vs:
            p = self._p(v, idx)
            o = np.argsort(p)
            c = np.cumsum(w[o])
            out.append([float(w @ p)] + [float(p[o][np.searchsorted(c, qq)]) for qq in q])
        return np.array(out)        # columns: mean, q05, q50, q95

    def recommend(self, candidates, target, batch=20):
        """Speed whose next batch most reduces the summed posterior variance of S0(target)."""
        idx, w = self._active()
        E = np.stack([self._p(t, idx) for t in target], axis=1)
        mu = w @ E
        var0 = float((w @ (E - mu) ** 2).sum())
        ks = np.arange(batch + 1)
        logc = special.gammaln(batch + 1) - special.gammaln(ks + 1) - special.gammaln(batch - ks + 1)
        gains = []
        for x in candidates:
            p = np.clip(self._p(x, idx), 1e-9, 1 - 1e-9)
            pmf = np.exp(logc[None, :] + ks[None, :] * np.log(p)[:, None] + (batch - ks)[None, :] * np.log1p(-p)[:, None])
            marg = w @ pmf
            W = (w[:, None] * pmf) / np.clip(marg[None, :], 1e-300, None)
            mk, m2 = W.T @ E, W.T @ E ** 2
            gains.append(var0 - float(marg @ (m2 - mk ** 2).sum(axis=1)))
        return float(candidates[int(np.argmax(gains))]), np.array(gains), math.sqrt(var0 / len(target))


# ----------------------------------------------------------------------- CLI
def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _point(text):
    v, k, n = (float(x) for x in text.split(","))
    return v, int(k), int(n)


def _probe(text):
    line, v, k, n = text.split(",")
    parse_line(line)
    return line.strip().lower(), float(v), int(k), int(n)


def _validate(args):
    from analysis.latency_model import refined
    return refined.main(variants=args.variants.split(","), policy=args.policy)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("csv")
    f.add_argument("--horizon", type=int, default=H)
    f.add_argument("--no-prior", action="store_true", help="plain maximum likelihood (no prior from earlier tasks)")
    f.add_argument("--c", type=float, default=0.0, help=f"fixed plan-tail constant (pi0.5: {C_PI05})")
    f.add_argument("--fit-c", action="store_true", help="fit the plan-tail constant from this task's lines")
    f.add_argument("--gamma", action="store_true", help="also fit the interception term of the synchronous lines")
    nx = sub.add_parser("next")
    nx.add_argument("csv", help="columns v,k,n (sync 0 ms episodes so far); may be empty")
    nx.add_argument("--vmax", type=float, default=0.2, help="largest speed you will report")
    nx.add_argument("--batch", type=int, default=20)
    nx.add_argument("--grid", type=str, default=None, help="comma-separated speeds to report (default 0.001 + 16 steps to vmax)")
    pr = sub.add_parser("predict")
    for name in ("s", "v50", "beta", "f", "a", "b"):
        pr.add_argument(f"--{name}", type=float, required=True)
    pr.add_argument("--c", type=float, default=0.0, help=f"plan-tail constant (pi0.5: {C_PI05})")
    pr.add_argument("--gamma", type=float, default=0.0, help="interception term (synchronous lines only)")
    pr.add_argument("--mode", default="sync")
    pr.add_argument("--latency", type=float, default=0.0)
    pr.add_argument("--horizon", type=int, default=H)
    pr.add_argument("--vmax", type=float, default=0.2)
    pb = sub.add_parser("probe")
    pb.add_argument("csv", help="columns v,k,n (sync 0 ms episodes)")
    pb.add_argument("--latency", type=float, default=200.0, help="latency of both probes (ms)")
    pb.add_argument("--lazy", type=_point, help="lazy probe as v,k,n")
    pb.add_argument("--sync", type=_point, help="synchronous probe as v,k,n")
    pb.add_argument("--horizon", type=int, default=H)
    pb.add_argument("--no-prior", action="store_true", help="plain maximum likelihood for S0")
    tr = sub.add_parser("transfer")
    tr.add_argument("csv", help="columns v,k,n (the complete reference line)")
    tr.add_argument("--line", required=True, help="setting of the complete line: lazy100, sync150, cont200, ...")
    tr.add_argument("--probe", type=_probe, action="append", default=[],
                    help="one point on another line as line,v,k,n; give two, so that the three lines include "
                         "a synchronous and an asynchronous one")
    tr.add_argument("--horizon", type=int, default=H)
    va = sub.add_parser("validate")
    va.add_argument("--variants", default="M1", help="comma-separated: M1 (a, b), M2 (+ c), M4 (+ gamma)")
    va.add_argument("--policy", choices=("pi0.5", "MolmoAct2"), default="pi0.5",
                    help="policy whose four LIBERO tasks are tested (both ran 50/150/250 ms)")
    args = ap.parse_args(argv)

    if args.cmd == "fit":
        rows = [(r["line"], r["v"], r["k"], r["n"]) for r in _read(args.csv)]
        lines = {r[0] for r in rows}
        if len(lines) > 1 and not separates_age_and_pause(lines):
            print("warning: these lines cannot separate a from b (need a synchronous and an asynchronous line at a "
                  "nonzero delay); a and b below rest on the prior")
        res = fit(rows, horizon=args.horizon, fit_ab=len(lines) > 1,
                  prior=None if args.no_prior else POP_PRIOR, c=args.c, fit_c=args.fit_c, fit_gamma=args.gamma)
        print(f"S0: s={res['s']:.3f}  v50={res['v50']:.4f} m/s  beta={res['beta']:.2f}  floor={res['f']:.3f}")
        print(f"sensitivities: a={res['a']:.3f} per control step of extra observation age, b={res['b']:.3f} per unit pause ratio")
        if res["c"] or res["gamma"]:
            print(f"plan-tail constant c={res['c']:.3f}, interception gamma={res['gamma']:.2f}")
        print(f"fit RMSE: {res['rmse_pp']:.2f} pp over {len(rows)} points")
        for line in sorted({r[0] for r in rows}):
            m, L = parse_line(line)
            rho = plan_tail(m, L, res["c"], args.horizon)
            print(f"  {line:8s} kappa={kappa(m, L, res['a'], res['b'], args.horizon):.3f}" + (f"  rho={rho:.3f}" if rho < 1 else ""))
    elif args.cmd == "next":
        post = AdaptiveS0()
        n_total = 0
        for r in _read(args.csv):
            post.update(float(r["v"]), float(r["k"]), float(r["n"]))
            n_total += int(float(r["n"]))
        grid = (np.array([float(x) for x in args.grid.split(",")]) if args.grid
                else np.concatenate([[0.001], np.linspace(args.vmax / 16, args.vmax, 16)]))
        cands = np.concatenate([[0.001], np.exp(np.linspace(np.log(0.005), np.log(args.vmax), 20))])
        x, gains, sd = post.recommend(cands, grid, args.batch)
        b = post.band(grid)
        print(f"episodes so far: {n_total}; posterior RMS sd of the curve: {sd * 100:.1f} pp")
        print(" speed   mean   5%    95%")
        for v, row in zip(grid, b):
            print(f" {v:6.3f}  {row[0] * 100:5.1f}  {row[1] * 100:5.1f}  {row[3] * 100:5.1f}")
        print(f"next batch: {args.batch} episodes at v = {x:.4f} m/s")
    elif args.cmd == "predict":
        vs = np.concatenate([[0.001], np.linspace(args.vmax / 8, args.vmax, 8)])
        p = predict(vs, args.mode, args.latency, args.s, args.v50, args.beta, args.f, args.a, args.b, args.horizon,
                    c=args.c, gamma=args.gamma)
        print(f"kappa={kappa(args.mode, args.latency, args.a, args.b, args.horizon):.3f}")
        for v, q in zip(vs, p):
            print(f" {v:6.3f}  {q * 100:5.1f}%")
    elif args.cmd == "probe":
        rows = [("sync0", r["v"], r["k"], r["n"]) for r in _read(args.csv)]
        res = fit(rows, horizon=args.horizon, fit_ab=False, prior=None if args.no_prior else POP_PRIOR)
        s0 = (res["s"], res["v50"], res["beta"], res["f"])
        print(f"S0: s={res['s']:.3f}  v50={res['v50']:.4f} m/s  beta={res['beta']:.2f}  floor={res['f']:.3f}")
        print(f"probe speed: v50 / {PROBE_FACTOR} = {res['v50'] / PROBE_FACTOR:.4f} m/s; run one lazy and one synchronous "
              f"batch there at {args.latency:g} ms")
        if args.lazy and args.sync:
            out = probe(s0, args.lazy, args.sync, args.latency, horizon=args.horizon)
            print(f"probes act like kappa: lazy {out['kappa_lazy']:.3f}, synchronous {out['kappa_sync']:.3f}")
            print(f"sensitivities: a={out['a']:.3f} ± {out['a_sd']:.3f}, b={out['b']:.3f} ± {out['b_sd']:.3f} "
                  f"(sampling spread of the two probes)")
            for mode in ("sync", "lazy", "cont"):
                print(f"  {mode}: " + "  ".join(f"{lat} ms kappa={kappa(mode, lat, out['a'], out['b'], args.horizon):.3f}"
                                               for lat in (50, 100, 150, 200, 250)))
    elif args.cmd == "transfer":
        ref = [(r["v"], r["k"], r["n"]) for r in _read(args.csv)]
        out = transfer(ref, args.line, args.probe, horizon=args.horizon, draws=300 if args.probe else 0)
        r = out["reference"]
        print(f"{args.line}: measured at {len(r.v)} speeds up to {r.v[-1]:g} m/s; midpoint speed {r.midpoint_speed():.4f} m/s")
        print(f"probe speed: midpoint / {PROBE_FACTOR} = {r.midpoint_speed() / PROBE_FACTOR:.4f} m/s; the three lines need "
              f"at least one synchronous and one asynchronous line at a nonzero delay")
        if len(args.probe) >= 2:
            print(f"sensitivities: a={out['a']:.3f} ± {out['a_sd']:.3f}, b={out['b']:.3f} ± {out['b_sd']:.3f} "
                  f"(sampling spread of the probes); base line kappa {out['kappa_ref']:.3f}")
            print("zero-delay curve: " + "  ".join(
                f"{v:g} {100 * float(predict_from(r, v, 'sync', 0, out['a'], out['b'], args.horizon, ref_feat=out['ref_feat'])[0]):.1f}%"
                for v in r.v))
            for mode in ("sync", "lazy", "cont"):
                print(f"  {mode}: " + "  ".join(f"{lat} ms kappa={kappa(mode, lat, out['a'], out['b'], args.horizon):.3f}"
                                               for lat in (50, 100, 150, 200, 250)))
        elif args.probe:
            print("one probe fixes only one combination of a and b; add a second probe on another line")
    elif args.cmd == "validate":
        _validate(args)


if __name__ == "__main__":
    main()

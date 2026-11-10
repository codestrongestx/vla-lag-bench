"""The sync-0 curve family, binomial fitting, and the error metrics shared by the analyses."""
import numpy as np
from scipy import optimize, special

from analysis.latency_model.speed_latency_model import s0_curve


def nll_binom(k, n, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(k * np.log(p) + (n - k) * np.log1p(-p))


def deviance(k, n, p):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(k / (n * p)), 0.0)
        t2 = np.where(n - k > 0, (n - k) * np.log((n - k) / (n * (1 - p))), 0.0)
    return 2 * (t1 + t2)


def rmse_pp(k, n, p):
    return float(np.sqrt(np.mean((np.asarray(k) / np.asarray(n) - np.asarray(p)) ** 2)) * 100)


def noise_pp(k, n):
    """RMS binomial standard error of the observed points (pp): what a perfect model would score."""
    p = np.asarray(k) / np.asarray(n)
    return float(np.sqrt(np.mean(p * (1 - p) / np.asarray(n))) * 100)


def unpack_s0(x):
    return special.expit(x[0]), float(np.exp(x[1])), float(np.exp(x[2])), special.expit(x[3])


S0_BOUNDS = [(-6.0, 9.0), (np.log(1e-3), np.log(5.0)), (np.log(0.2), np.log(30.0)), (-9.0, 3.0)]
S0_STARTS = [[3.0, lv, lb, -1.5] for lv in (np.log(0.04), np.log(0.1), np.log(0.2)) for lb in (np.log(2.0), np.log(4.5))]


def minimize_multi(fun, starts, bounds):
    best = None
    for x0 in starts:
        r = optimize.minimize(fun, np.asarray(x0, float), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    r = optimize.minimize(fun, best.x, method="Powell", bounds=bounds,
                          options=dict(maxiter=40000, xtol=1e-8, ftol=1e-12))
    return r if r.fun < best.fun else best


def fit_s0(points):
    """Maximum-likelihood sync-0 curve (s, v50, beta, f) of one task's synchronous 0 ms points."""
    v, k, n = points.v.astype(float), points.k.astype(float), points.n.astype(float)
    r = minimize_multi(lambda x: nll_binom(k, n, s0_curve(v, *unpack_s0(x))).sum(), S0_STARTS, S0_BOUNDS)
    return unpack_s0(r.x)

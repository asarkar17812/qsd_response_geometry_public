"""Exact e-flatness with properly non-uniform killing and a response image of dimension >= 2.

This realizes Example 5.10 and Proposition 5.11. The two earlier examples each exercise only half of
the claim: Example 5.8 is e-flat but uniform-killing (h=1), and Example 5.9 is non-uniform (h!=1) but
has a one-dimensional image (so flatness is vacuous). A single tilted-reset construction is
simultaneously e-flat, properly non-uniform, and dimension-2.

THE CONSTRUCTION (a *tilted reset*). Fix a reference pi, observables f_1,...,f_k linearly
independent mod constants, and a FIXED non-uniform killing vector kappa >= 0 (not constant).
For theta in R^k put the exponential family
    nu_theta(s) = pi_s exp(sum_a theta^a f_a(s)) / Z(theta),
let Lambda(theta) = nu_theta . kappa (the QSD identity Lambda = nu.q), and the TILTED reset
distribution
    m_theta(s) = nu_theta(s) (1 + kappa_s - Lambda(theta)).
Then
    Q(theta) = 1 m_theta - I - diag(kappa).
Q is an irreducible Metzler sub-generator (off-diagonals m_theta(s') > 0), with killing
q = kappa (properly non-uniform), and -- by the tilt -- nu_theta is EXACTLY its quasi-stationary
distribution with decay rate Lambda(theta):
    nu_theta Q = m_theta - nu_theta - nu_theta*kappa = -Lambda nu_theta.
Example 5.8 is the special case kappa = Lambda*1 (uniform), where m_theta = nu_theta and the
reset is untilted. The non-uniform tilt makes h != 1 and the normalization correction c_j
active, while the response image remains the exponential family -- e-flat with
pullback Fisher = Cov_{nu}(f_a, f_b).

We verify, over many random theta: nu matches the exponential family; -Lambda is dominant;
Q is Metzler; h != 1; the pullback Fisher equals Cov(f_a,f_b) (e-flatness); the response formula
eq.(5) with c_j holds; and c_j is active (the killed-case machinery is genuinely exercised).

Run:  python eflat_nonuniform_qsd.py
Outputs: output/eflat_nonuniform_qsd.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import config

# calibrated instance (the one printed in the paper)
PI = np.array([0.25, 0.25, 0.25, 0.25])
F1 = np.array([0.0, 1.0, 0.0, 1.0])
F2 = np.array([0.0, 0.0, 1.0, 1.0])
KAPPA = np.array([0.10, 0.20, 0.15, 0.25])
N = 4


def nu_of(theta):
    w = PI * np.exp(theta[0] * F1 + theta[1] * F2)
    return w / w.sum()


def make_Q(theta):
    nu = nu_of(theta)
    Lam = float(nu @ KAPPA)
    m = nu * (1 + KAPPA - Lam)
    return np.outer(np.ones(N), m) - np.eye(N) - np.diag(KAPPA)


def principal(Q):
    w, vr = np.linalg.eig(Q)
    k = int(np.argmax(w.real))
    Lam = -w[k].real
    h = vr[:, k].real
    wl, vl = np.linalg.eig(Q.T)
    kl = int(np.argmax(wl.real))
    nu = vl[:, kl].real
    if nu.sum() < 0:
        nu = -nu
    nu = nu / nu.sum()
    if h @ nu < 0:
        h = -h
    h = h / (nu @ h)
    A = Q + Lam * np.eye(N)
    P = np.outer(h, nu)
    R = np.linalg.solve(A + P, np.eye(N) - P)
    return Lam, nu, h, R


def _fd(f, t, eps=1e-6):
    p = len(t)
    f0 = np.atleast_1d(f(t))
    J = np.zeros((len(f0), p))
    for j in range(p):
        tp = t.copy(); tp[j] += eps
        tm = t.copy(); tm[j] -= eps
        J[:, j] = (np.atleast_1d(f(tp)) - np.atleast_1d(f(tm))) / (2 * eps)
    return J


def verify(n_random=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    agg = {"nu_match": 0.0, "fisher_minus_cov": 0.0, "dnu_err": 0.0,
           "min_max_h_minus_1": 1e9, "max_abs_cj": 0.0,
           "always_dominant": True, "always_metzler": True, "always_rankJ2": True}
    for _ in range(n_random):
        t = rng.uniform(-2.0, 2.0, size=2)
        Q = make_Q(t)
        Lam, nu, h, R = principal(Q)
        one = np.ones(N)
        agg["nu_match"] = max(agg["nu_match"], float(np.max(np.abs(nu - nu_of(t)))))
        agg["min_max_h_minus_1"] = min(agg["min_max_h_minus_1"], float(np.max(np.abs(h - one))))
        off = Q - np.diag(np.diag(Q))
        agg["always_metzler"] = agg["always_metzler"] and bool(np.all(off >= -1e-12))
        evs = np.linalg.eigvals(Q)
        agg["always_dominant"] = agg["always_dominant"] and bool(np.isclose(np.max(evs.real), -Lam))
        dQ = _fd(lambda tt: make_Q(tt).reshape(-1), t).reshape(N, N, 2)
        cj = np.array([nu @ dQ[:, :, j] @ R @ one for j in range(2)])
        agg["max_abs_cj"] = max(agg["max_abs_cj"], float(np.max(np.abs(cj))))
        Jfd = _fd(lambda tt: principal(make_Q(tt))[1], t)
        Jan = np.zeros((N, 2))
        for j in range(2):
            Jan[:, j] = -nu @ dQ[:, :, j] @ R + cj[j] * nu
        agg["dnu_err"] = max(agg["dnu_err"], float(np.max(np.abs(Jfd - Jan))))
        agg["always_rankJ2"] = agg["always_rankJ2"] and bool(
            np.sum(np.linalg.svd(Jan, compute_uv=False) > 1e-7) == 2)
        G = Jan.T @ np.diag(1.0 / nu) @ Jan
        Ef1, Ef2 = nu @ F1, nu @ F2
        Cov = np.array([[np.sum(nu * (F1 - Ef1) ** 2), np.sum(nu * (F1 - Ef1) * (F2 - Ef2))],
                        [np.sum(nu * (F1 - Ef1) * (F2 - Ef2)), np.sum(nu * (F2 - Ef2) ** 2)]])
        agg["fisher_minus_cov"] = max(agg["fisher_minus_cov"], float(np.max(np.abs(G - Cov))))
    return agg


def calibrated_point(theta=(0.7, -0.5)) -> dict:
    t = np.array(theta)
    Q = make_Q(t)
    Lam, nu, h, R = principal(Q)
    return {"theta": list(theta), "kappa": KAPPA.tolist(), "q_equals_kappa": True,
            "Lambda": round(Lam, 6), "nu": [round(x, 5) for x in nu],
            "h": [round(x, 5) for x in h], "max_h_minus_1": round(float(np.max(np.abs(h - 1))), 5),
            "spectrum_real": sorted([round(float(x), 5) for x in np.linalg.eigvals(Q).real],
                                    reverse=True)}


def run() -> dict:
    v = verify()
    cp = calibrated_point()
    result = {
        "experiment": "exact e-flat + properly non-uniform killing + dimension-2 image (Example 5.10, Proposition 5.11)",
        "construction": "tilted reset Q = 1 m_theta - I - diag(kappa), m_theta = nu_theta*(1+kappa-Lambda)",
        "calibrated_point": cp,
        "verification_over_random_theta": v,
        "verdict": (
            "The tilted-reset construction is simultaneously (a) exactly e-flat -- the pullback "
            "Fisher equals Cov_nu(f_a,f_b) to %.1e over 2000 random theta; (b) properly non-uniform -- "
            "killing q=kappa is not proportional to 1, and min over theta of max|h-1| = %.3f > 0, so "
            "h != 1 everywhere; (c) dim-2 -- rank J = 2 throughout. nu matches the exponential family to "
            "%.1e, -Lambda is always the dominant eigenvalue, Q is always Metzler, the response formula "
            "eq.(5) holds to %.1e, and the normalization correction c_j is ACTIVE (max|c_j|=%.3f), so the "
            "killed-case machinery is genuinely exercised in the multi-dimensional regime. Example 5.8 is "
            "the untilted (kappa=Lambda*1) special case."
            % (v["fisher_minus_cov"], v["min_max_h_minus_1"], v["nu_match"], v["dnu_err"],
               v["max_abs_cj"])),
        "supports": "qsd_response_geometry.tex Section 5: Example 5.10 (e-flat non-uniform) and Proposition 5.11.",
    }
    os.makedirs(config.output_path(), exist_ok=True)
    with open(config.output_path("eflat_nonuniform_qsd.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

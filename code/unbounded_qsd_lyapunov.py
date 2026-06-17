"""Example 8.6: the birth-death-with-killing chain on a non-compact state space.

The active count n in {0, 1, 2, ...} evolves by birth n -> n+1 at rate beta, death n -> n-1 at rate
mu*n, and killing n -> cemetery at rate kappa*n. The living generator is

    (L f)(n) = beta (f(n+1) - f(n)) + mu*n (f(n-1) - f(n)) - kappa*n f(n).

For the weight W(n) = exp(gamma*n) the drift ratio
(L W / W)(n) = beta(e^gamma - 1) - n[mu(1 - e^{-gamma}) + kappa] is affine in n with strictly negative
slope, so the Foster-Lyapunov condition L W <= -c W + b 1_K holds with an explicit finite small set
K = {0, ..., N0 - 1}. The chain is exactly solvable: with rho = beta/(mu + kappa),

    nu = Poisson(rho),   Lambda = kappa*beta/(mu + kappa) = kappa*rho,   spectral gap = mu + kappa.

This script checks the closed forms against a deterministic truncation and regenerates Figure 4.
Outputs go under output/.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def constants(beta=1.0, mu=0.5, kappa=0.1, gamma=1.0, c=1.0):
    rho = beta / (mu + kappa)
    Lambda = kappa * rho
    gap = mu + kappa
    slope = mu * (1 - math.exp(-gamma)) + kappa
    intercept = beta * (math.exp(gamma) - 1)
    N0 = math.ceil((intercept + c) / slope)
    drift = lambda n: intercept - n * slope
    b = max((drift(n) + c) * math.exp(gamma * n) for n in range(N0))
    exp_moment = math.exp(rho * (math.e - 1))
    return dict(beta=beta, mu=mu, kappa=kappa, gamma=gamma, c=c, rho=rho, Lambda=Lambda, gap=gap,
                N0=N0, b=b, exp_moment=exp_moment, slope=slope, intercept=intercept)


def truncated_spectrum(beta, mu, kappa, M=600):
    """Two smallest decay rates and the Yaglom QSD of the living generator truncated to n = 0 .. M-1."""
    L = np.zeros((M, M))
    for n in range(M):
        if n + 1 < M:
            L[n, n + 1] += beta
            L[n, n] -= beta
        if n - 1 >= 0:
            L[n, n - 1] += mu * n
            L[n, n] -= mu * n
        L[n, n] -= kappa * n
    w, V = np.linalg.eig(L.T)
    order = np.argsort(w.real)[::-1]
    Lambda_N = -w[order[0]].real
    gap_N = (w[order[0]] - w[order[1]]).real
    nu = np.real(V[:, order[0]])
    nu = nu / nu.sum()
    EN = float((np.arange(M) * nu).sum())
    return Lambda_N, gap_N, EN


def _figure(k, out):
    n = np.arange(0, 14)
    drift = k["intercept"] - n * k["slope"]
    rho = k["rho"]
    nn = np.arange(0, 16)
    pois = np.exp(-rho) * rho ** nn / np.array([math.factorial(int(j)) for j in nn])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].axhline(-k["c"], color="0.6", ls="--", lw=1)
    ax[0].axvspan(k["N0"], 13, color="0.9")
    ax[0].plot(n, drift, "o-", color="#1f4e79")
    ax[0].set_xlabel("n"); ax[0].set_ylabel(r"$D(n) = (LW/W)(n)$")
    ax[0].set_title("drift ratio (affine, negative slope)")
    ax[1].semilogy(nn, pois, "s-", color="#7a1f1f")
    ax[1].set_xlabel("n"); ax[1].set_ylabel(r"$\nu(n)$")
    ax[1].set_title(r"quasi-stationary law: Poisson($\rho$)")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "output", "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "output", "figures", "unbounded_qsd_lyapunov.png"), dpi=150)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run():
    k = constants()
    Lambda_N, gap_N, EN = truncated_spectrum(k["beta"], k["mu"], k["kappa"])
    result = {
        "instance": "birth-death-with-killing chain (active count), Example 8.6",
        "calibrated": {key: k[key] for key in ("beta", "mu", "kappa", "gamma", "c")},
        "closed_form": {
            "rho": k["rho"], "Lambda": k["Lambda"], "gap": k["gap"], "N0": k["N0"],
            "b": round(k["b"], 4), "exp_moment": round(k["exp_moment"], 6),
        },
        "truncation_check": {
            "Lambda_truncated": Lambda_N,
            "Lambda_abs_err": abs(Lambda_N - k["Lambda"]),
            "gap_truncated": gap_N,
            "gap_abs_err": abs(gap_N - k["gap"]),
            "identity_Lambda_eq_kappa_E_nu_N": k["kappa"] * EN,
            "identity_abs_err": abs(k["kappa"] * EN - k["Lambda"]),
        },
    }
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    with open(os.path.join(HERE, "output", "unbounded_qsd_lyapunov.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _figure(k, os.path.join(HERE, "output", "unbounded_qsd_lyapunov.png"))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

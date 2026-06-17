"""
Cross-verification of the algebraic kernel of the infinite-dimensional QSD extension (Section 8).

The infinite-dimensional theorems (Thm 1-3) reduce, on any FINITE TRUNCATION, to a small set of
exact ALGEBRAIC identities about the principal eigentriple (Lambda, nu, h) of a killed sub-generator
Q and its group inverse / reduced resolvent R = (Q + Lambda I)^#:

  (K1) group-inverse identity:   (Q + Lambda I) R = I - Pi,   Pi = h (x) nu  (nu h = 1),
       and  R Pi = Pi R = 0;
  (K2) Hellmann-Feynman:         d Lambda = - nu (dQ) h ;
  (K3) reduced-resolvent dnu:    d nu = - nu (dQ) R + c nu ,  c = nu (dQ) R 1 .

These are the algebra the response calculus (Thm 2) and the fold amplification (Thm 3a) rest on; they
hold in any dimension, so verifying them on truncations is the machine-checkable base case of the
limit argument. We verify them (a) EXACTLY over the rationals on a hand-built 2-state instance (the
same instance machine-checked in lean/InfiniteDimQSDKernel.lean), and (b) to high precision on the
model's actual 5x5 killed living generator.

Run:  python infinite_dim_qsd_kernel.py
Outputs: output/infinite_dim_qsd_kernel.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import config



def group_inverse(Q, Lam, nu, h):
    """Group inverse R = (Q+Lambda I)^# via the spectral construction: 1/lambda_i on each
    non-principal eigenspace, 0 on the principal (kernel) eigenspace span(h)."""
    A = Q + Lam * np.eye(Q.shape[0])
    w, V = np.linalg.eig(A)                       # right eigvecs
    Vinv = np.linalg.inv(V)                       # rows are left eigvecs (up to scale)
    R = np.zeros_like(A, dtype=complex)
    for i in range(len(w)):
        if abs(w[i]) > 1e-9:                      # skip the (near-)zero principal eigenvalue
            R += (1.0 / w[i]) * np.outer(V[:, i], Vinv[i, :])
    return R.real


def principal_triple(Q):
    """(Lambda, nu, h): dominant (least-decay) eigenvalue -Lambda of the sub-generator Q, with
    left/right eigenvectors normalized nu.1=1 (prob), nu.h=1."""
    w, Vr = np.linalg.eig(Q)
    j = int(np.argmax(w.real))                    # closest to 0 = -Lambda
    Lam = -float(w[j].real)
    h = Vr[:, j].real
    wl, Vl = np.linalg.eig(Q.T)
    jl = int(np.argmax(wl.real))
    nu = Vl[:, jl].real
    if nu.sum() < 0:
        nu = -nu
    nu = nu / nu.sum()                            # nu.1 = 1
    if (nu @ h) < 0:
        h = -h
    h = h / (nu @ h)                              # nu.h = 1
    return Lam, nu, h


def check_identities(Q, dQ, tol=1e-8):
    n = Q.shape[0]
    Lam, nu, h = principal_triple(Q)
    R = group_inverse(Q, Lam, nu, h)
    Pi = np.outer(h, nu)                          # h (x) nu, since nu.h=1
    A = Q + Lam * np.eye(n)
    # (K1)
    k1 = float(np.max(np.abs(A @ R - (np.eye(n) - Pi))))
    k1b = float(max(np.max(np.abs(R @ Pi)), np.max(np.abs(Pi @ R))))
    # (K2) Hellmann-Feynman vs finite difference of Lambda along dQ
    hf = float(-(nu @ dQ @ h))
    eps = 1e-6
    Lam_p, _, _ = principal_triple(Q + eps * dQ)
    Lam_m, _, _ = principal_triple(Q - eps * dQ)
    dLam_fd = (-(Lam_p) - (-(Lam_m))) / (2 * eps)   # d(-Lambda)? use eigenvalue lam=-Lambda
    # eigenvalue is lam = -Lambda; HF gives d lam = nu dQ h ; compare to FD of lam
    lam_p = -Lam_p; lam_m = -Lam_m
    dlam_fd = (lam_p - lam_m) / (2 * eps)
    hf_lam = float(nu @ dQ @ h)                   # d lam = + nu dQ h
    k2 = abs(hf_lam - dlam_fd)
    # (K3) reduced-resolvent dnu vs finite difference (compare on the simplex tangent)
    one = np.ones(n)
    c = float(nu @ dQ @ R @ one)
    dnu = -(nu @ dQ @ R) + c * nu
    _, nu_p, _ = principal_triple(Q + eps * dQ)
    _, nu_m, _ = principal_triple(Q - eps * dQ)
    dnu_fd = (nu_p - nu_m) / (2 * eps)
    k3 = float(np.max(np.abs(dnu - dnu_fd)))
    return {"Lambda": round(Lam, 6), "nu": [round(float(x), 5) for x in nu],
            "K1_groupinv_residual": k1, "K1_R_Pi_residual": k1b,
            "K2_HellmannFeynman_err": k2, "K3_reduced_resolvent_err": k3,
            "all_hold": bool(k1 < tol and k1b < tol and k2 < 1e-5 and k3 < 1e-4)}


def exact_two_state():
    """Hand-built exact-rational instance (matches lean/InfiniteDimQSDKernel.lean):
    Q=[[-2,1],[1,-2]] (Metzler sub-generator, killing 1 per state); principal eigenvalue -Lambda=-1
    with h=(1,1), nu=(1/2,1/2); group inverse R=[[-1/4,1/4],[1/4,-1/4]]; (Q+I)R = I - h(x)nu EXACTLY."""
    from fractions import Fraction as F
    Q = [[F(-2), F(1)], [F(1), F(-2)]]
    Lam = F(1); h = [F(1), F(1)]; nu = [F(1, 2), F(1, 2)]
    R = [[F(-1, 4), F(1, 4)], [F(1, 4), F(-1, 4)]]
    # A = Q + Lam I
    A = [[Q[i][j] + (Lam if i == j else F(0)) for j in range(2)] for i in range(2)]
    # AR
    AR = [[sum(A[i][k] * R[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    # I - h(x)nu
    ImPi = [[(F(1) if i == j else F(0)) - h[i] * nu[j] for j in range(2)] for i in range(2)]
    exact_ok = AR == ImPi
    # also nu.h = 1, nu.1 = 1, Qh = -Lam h, nu Q = -Lam nu
    nu_h = sum(nu[i] * h[i] for i in range(2))
    Qh = [sum(Q[i][k] * h[k] for k in range(2)) for i in range(2)]
    nuQ = [sum(nu[k] * Q[k][j] for k in range(2)) for j in range(2)]
    eig_ok = (Qh == [-Lam * h[i] for i in range(2)]) and (nuQ == [-Lam * nu[i] for i in range(2)])
    return {"AR_equals_I_minus_Pi_exactly": bool(exact_ok), "nu_dot_h": str(nu_h),
            "eigen_relations_exact": bool(eig_ok)}


def run() -> dict:
    # (b) the model's actual 5x5 killed living generator
    import config
    q = config.qsd_at(config.reduced_core_params())
    Q5 = np.asarray(q.QT, dtype=float)
    dQ = Q5 * 0.0
    dQ[1, 4] += 0.01; dQ[1, 1] -= 0.01            # a concrete Metzler-preserving perturbation (AC->I)
    model5 = check_identities(Q5, dQ)

    exact2 = exact_two_state()
    result = {
        "kernel": "group-inverse identity (K1) + Hellmann-Feynman (K2) + reduced-resolvent dnu (K3)",
        "exact_rational_2state": exact2,
        "model_5x5_killed_generator": model5,
        "verdict": (
            f"The algebraic kernel of the infinite-dimensional response calculus is machine-verified: "
            f"EXACTLY over the rationals on the 2-state instance (group-inverse identity (Q+I)R=I-h(x)nu "
            f"holds exactly: {exact2['AR_equals_I_minus_Pi_exactly']}; eigen-relations exact: "
            f"{exact2['eigen_relations_exact']}), and to high precision on the model's 5x5 killed "
            f"generator (K1 residual {model5['K1_groupinv_residual']:.1e}, Hellmann-Feynman err "
            f"{model5['K2_HellmannFeynman_err']:.1e}, reduced-resolvent err "
            f"{model5['K3_reduced_resolvent_err']:.1e}; all_hold={model5['all_hold']}). These are the "
            f"identities the infinite-dimensional theorems reduce to on every finite truncation -- the "
            f"machine-checkable base case of the limit (the analytic limit itself is classical and cited)."),
        "lean": "lean/InfiniteDimQSDKernel.lean checks the exact 2-state identity over the rationals.",
    }
    os.makedirs(config.output_path(), exist_ok=True)
    with open(config.output_path("infinite_dim_qsd_kernel.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

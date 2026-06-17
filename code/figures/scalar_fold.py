"""
Figure B (closed-form scalar fold) for "Quasi-Stationary Response Geometry".

Example 6.5 / Theorem 6.4.  Scalar self-consistency F(theta,u)=u^2-theta has the
regular branch u*(theta)=sqrt(theta) with smallest singular value sigma=|2u*|=
2 sqrt(theta).  Coupling u to the two-state killed chain through the activation rate
a(u)=a0+alpha u, the closed-loop QSD is nu*(theta)=nu(a0+alpha sqrt(theta)), and the
closed-loop Fisher metric (one parameter)

    G*(theta) = sum_s (d nu*_s/d theta)^2 / nu*_s

is verified to blow up as G* ~ C/theta ~ C' sigma^{-2}, with the closed-form leading
coefficient C = (alpha^2/4) sum_s (d_a nu_s)^2 / nu_s evaluated at a=a0.

Deterministic; writes a vector PDF.  Run:  python scalar_fold.py
"""
import numpy as np


def nu_of_a(a, b, kappa):
    """QSD (left Perron, prob row) of Q(a,b,kappa)=[[-(a+kappa),a],[b,-b]]."""
    Q = np.array([[-(a + kappa), a], [b, -b]])
    wl, vl = np.linalg.eig(Q.T)
    nu = vl[:, int(np.argmax(wl.real))].real
    if nu.sum() < 0:
        nu = -nu
    return nu / nu.sum()


# fixed chain parameters and coupling
a0, b, kappa, alpha = 1.0, 0.8, 0.3, 0.7

# closed-form leading coefficient  C = (alpha^2/4) sum_s (d_a nu_s)^2 / nu_s  at a0
da = 1e-6
nu_a0 = nu_of_a(a0, b, kappa)
d_a_nu = (nu_of_a(a0 + da, b, kappa) - nu_of_a(a0 - da, b, kappa)) / (2 * da)
C = (alpha ** 2 / 4.0) * np.sum(d_a_nu ** 2 / nu_a0)
print(f"a0={a0}, b={b}, kappa={kappa}, alpha={alpha}")
print(f"leading coefficient C = (alpha^2/4) sum (d_a nu)^2/nu |_a0 = {C:.6f}")

# closed-loop Fisher metric along the branch
thetas = np.geomspace(1e-6, 1e-1, 60)
Gstar = np.zeros_like(thetas)
for i, th in enumerate(thetas):
    a = a0 + alpha * np.sqrt(th)
    nu = nu_of_a(a, b, kappa)
    dth = 1e-7 * max(th, 1e-3)
    nu_p = nu_of_a(a0 + alpha * np.sqrt(th + dth), b, kappa)
    nu_m = nu_of_a(a0 + alpha * np.sqrt(max(th - dth, 1e-12)), b, kappa)
    dnu_dth = (nu_p - nu_m) / (2 * dth)
    Gstar[i] = np.sum(dnu_dth ** 2 / nu)

sigma = 2.0 * np.sqrt(thetas)

# G* = C/theta + O(theta^{-1/2}), so the leading -1 (in theta) / -2 (in sigma) slope is the
# theta -> 0 ASYMPTOTE; the O(theta^{-1/2}) correction biases a full-range fit. Fit on the
# asymptotic small-theta window (theta <= 1e-3, ~3 decades) where C/theta dominates.
asy = thetas <= 1e-3
slope_theta = np.polyfit(np.log(thetas[asy]), np.log(Gstar[asy]), 1)[0]
slope_sigma = np.polyfit(np.log(sigma[asy]), np.log(Gstar[asy]), 1)[0]
print(f"fitted slope  G* vs theta : {slope_theta:+.4f}  (predicted -1; window theta<=1e-3)")
print(f"fitted slope  G* vs sigma : {slope_sigma:+.4f}  (predicted -2; window theta<=1e-3)")
print(f"theta*G* -> {(thetas*Gstar)[0]:.6f} (should approach C={C:.6f})")

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 11,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

ax1.loglog(thetas, Gstar, "o", color="#1f3b73", ms=4, label=r"$\mathcal{G}^{*}(\theta)$")
ax1.loglog(thetas, C / thetas, "-", color="#c1121f", lw=1.6,
           label=r"$C/\theta$ (closed form)")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$\mathcal{G}^{*}(\theta)$")
ax1.set_title(fr"(a) $\mathcal{{G}}^{{*}}\sim C/\theta$ (fitted slope ${slope_theta:.2f}$)")
ax1.legend(frameon=False)
ax1.grid(alpha=0.25, which="both")

ax2.loglog(sigma, Gstar, "o", color="#1f3b73", ms=4, label=r"$\mathcal{G}^{*}$")
ax2.loglog(sigma, Gstar[-1] * (sigma / sigma[-1]) ** (-2.0), "--", color="gray", lw=1.0,
           label=r"Slope $-2$")
ax2.set_xlabel(r"$\sigma = 2\sqrt{\theta}$")
ax2.set_ylabel(r"$\mathcal{G}^{*}$")
ax2.set_title(fr"(b) $\mathcal{{G}}^{{*}}\sim\sigma^{{-2}}$ (fitted slope ${slope_sigma:.2f}$)")
ax2.legend(frameon=False)
ax2.grid(alpha=0.25, which="both")

# ----------------------------------------------------------------------
# GUARDRAIL: refuse to emit unless the predicted fold scalings hold (mirrors
# fold_example.py): G* ~ 1/theta (slope -1), G* ~ sigma^{-2} (slope -2), and the
# leading coefficient theta*G* -> C.
TOL = 0.12
ok = (abs(slope_theta + 1) < TOL) and (abs(slope_sigma + 2) < TOL) \
    and (abs((thetas * Gstar)[0] / C - 1) < 0.05)
print(f"GUARDRAIL: slopes (-1,-2) and theta*G*->C match within tol: {ok}")
if not ok:
    print(f"  STOPPING WITHOUT producing a figure: slope_theta={slope_theta:.3f} (want -1), "
          f"slope_sigma={slope_sigma:.3f} (want -2), theta*G*={(thetas*Gstar)[0]:.5f} (want C={C:.5f}).")
    raise SystemExit(2)

fig.tight_layout()
fig.savefig("scalar_fold.pdf")
print("FIGURE WRITTEN: scalar_fold.pdf")

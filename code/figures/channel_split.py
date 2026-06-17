"""
Figure A (channel split) for "Quasi-Stationary Response Geometry".

Two-state killed chain Q(a,b,kappa) = [[-(a+kappa), a],[b,-b]] (killing kappa out of
state 1). The QSD channel (response of nu) and the decay-rate channel (response of
Lambda) have distinct kernels (Proposition "channel split" / Definition "augmented
metric"). We exhibit two parameter directions:

  (a) the scaling gauge  Q -> s Q  : nu is exactly invariant, Lambda scales linearly
      -- a perturbation in ker(J_nu) but not in ker(grad Lambda);
  (b) a decay-flat direction in ker(grad Lambda): Lambda is stationary to first
      order while nu moves -- a perturbation in ker(grad Lambda) but not ker(J_nu).

Deterministic; writes a vector PDF.  Run:  python channel_split.py
"""
import numpy as np
np.set_printoptions(precision=6, suppress=True)


def principal_data(a, b, kappa):
    """Lambda>0, nu (left Perron, prob row), h (right, nu.h=1) of the 2-state chain."""
    Q = np.array([[-(a + kappa), a], [b, -b]])
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
    return Lam, nu, h


# base operating point (all rates positive; killing from state 1)
a0, b0, k0 = 1.0, 0.8, 0.3
Lam0, nu0, h0 = principal_data(a0, b0, k0)

# gradient of Lambda w.r.t. (a,b,kappa) by central differences
eps = 1e-6
base = np.array([a0, b0, k0])
gradLam = np.zeros(3)
for j in range(3):
    p = base.copy(); p[j] += eps
    m = base.copy(); m[j] -= eps
    gradLam[j] = (principal_data(*p)[0] - principal_data(*m)[0]) / (2 * eps)

# (b) a decay-flat direction: orthogonalize a fixed vector against grad Lambda
ghat = gradLam / np.linalg.norm(gradLam)
v0 = np.array([1.0, -1.0, 0.0])
v = v0 - (v0 @ ghat) * ghat
v = v / np.linalg.norm(v)          # grad(Lambda) . v = 0  (Lambda flat to 1st order)
print(f"base: a={a0}, b={b0}, kappa={k0};  Lambda0={Lam0:.5f}, nu0={nu0}")
print(f"grad Lambda = {gradLam};  decay-flat dir v = {v}  (grad.v={gradLam@v:.2e})")

# ---- (a) scaling gauge Q -> s Q ----
svals = np.linspace(0.6, 1.6, 41)
Lam_s, dnu_s = [], []
for s in svals:
    L, nu, _ = principal_data(s * a0, s * b0, s * k0)
    Lam_s.append(L / Lam0)
    dnu_s.append(np.abs(nu - nu0).sum())     # L1 movement of nu
Lam_s = np.array(Lam_s); dnu_s = np.array(dnu_s)

# ---- (b) decay-flat direction (a,b,kappa) += t v ----
tvals = np.linspace(-0.25, 0.25, 41)
Lam_t, dnu_t = [], []
for t in tvals:
    a, b, k = base + t * v
    L, nu, _ = principal_data(a, b, k)
    Lam_t.append(abs(L - Lam0) / Lam0)        # relative movement of Lambda
    dnu_t.append(np.abs(nu - nu0).sum())
Lam_t = np.array(Lam_t); dnu_t = np.array(dnu_t)

print(f"scaling gauge:  max |Delta nu|_1 = {dnu_s.max():.2e} (nu fixed); "
      f"Lambda/Lambda0 in [{Lam_s.min():.3f},{Lam_s.max():.3f}]")
print(f"decay-flat dir: max |Delta Lambda|/Lambda0 = {Lam_t.max():.2e} (Lambda flat, 1st order); "
      f"max |Delta nu|_1 = {dnu_t.max():.3f}")

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

ax1.plot(svals, Lam_s, "-", color="#c1121f", lw=1.8, label=r"$\Lambda(s)/\Lambda_0$")
ax1.plot(svals, dnu_s, "-", color="#1f3b73", lw=1.8, label=r"$\|\nu(s)-\nu_0\|_1$")
ax1.axhline(0.0, color="gray", lw=0.6, ls=":")
ax1.set_xlabel(r"Scaling $s$ in $Q \mapsto sQ$")
ax1.set_title(r"(a) Scaling gauge: $\Lambda$ moves, $\nu$ fixed")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(alpha=0.25)

ax2.plot(tvals, Lam_t, "-", color="#c1121f", lw=1.8,
         label=r"$|\Lambda(t)-\Lambda_0|/\Lambda_0$")
ax2.plot(tvals, dnu_t, "-", color="#1f3b73", lw=1.8, label=r"$\|\nu(t)-\nu_0\|_1$")
ax2.set_xlabel(r"Displacement $t$ along a decay-flat direction $v$")
ax2.set_title(r"(b) Decay-flat direction: $\nu$ moves, $\Lambda$ fixed")
ax2.legend(frameon=False, loc="upper center")
ax2.grid(alpha=0.25)

# ----------------------------------------------------------------------
# GUARDRAIL: refuse to emit unless the channel split holds numerically (mirrors
# fold_example.py): under the scaling gauge nu is exactly invariant while Lambda
# scales linearly (so the scaling direction is in ker(J_nu) but not ker(grad
# Lambda)), and on the decay-flat direction Lambda is stationary to first order
# while nu genuinely moves.
ok = (dnu_s.max() < 1e-10)                       # nu invariant under Q -> sQ
ok = ok and (Lam_s.max() - Lam_s.min() > 0.5)    # Lambda scales (not in ker grad-Lambda)
ok = ok and (Lam_t.max() < 1e-1)                 # Lambda flat to 1st order on the decay-flat dir
ok = ok and (dnu_t.max() > 1e-2)                 # nu genuinely moves there (split non-vacuous)
print(f"GUARDRAIL: channel split holds (nu scale-invariant, Lambda scales, "
      f"Lambda-flat dir moves nu): {ok}")
if not ok:
    print(f"  STOPPING WITHOUT producing a figure: max|dnu|(scaling)={dnu_s.max():.1e}, "
          f"Lambda range={Lam_s.max()-Lam_s.min():.2f}, max|dLambda|(flat dir)={Lam_t.max():.1e}, "
          f"max|dnu|(flat dir)={dnu_t.max():.2e}.")
    raise SystemExit(2)

fig.tight_layout()
fig.savefig("channel_split.pdf")
print("FIGURE WRITTEN: channel_split.pdf")

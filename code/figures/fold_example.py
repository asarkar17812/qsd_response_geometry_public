"""
Worked instance for Example 7.4 (self-consistent three-state fold) of
"Quasi-Stationary Response Geometry".

We build the 3-state killed cycle Q(theta,u) of the paper, close it
self-consistently by F(theta,u) = nu_1(theta,u) - u, and verify NUMERICALLY,
using ONLY the response calculus of the paper (Hellmann-Feynman (4), reduced-
resolvent (5), group inverse R=(A+P)^{-1}(I-P)), that as the operating point
approaches a fold sigma = |d_u F| -> 0:

      ||dPhi*|| ~ sigma^-1 ,   |grad Lambda*| ~ sigma^-1 ,   lambda_max(G*) ~ sigma^-2 .

The script (a) first confirms bistability exists (S-shaped self-consistency curve),
(b) locates the fold, (c) fits the three exponents by log-log regression and PRINTS
them, and (d) ONLY IF the fitted exponents match -1,-1,-2 within tolerance, writes a
vector PDF figure. Otherwise it stops without producing a figure.

Run:  python fold_example.py
"""
import sys
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ----------------------------------------------------------------------
# fixed model constants (state 1 = index 0, "rest"; 2 = index 1; 3 = index 2)
# killing kappa_1 out of state 1; cycle rates beta, gamma, delta.
# ----------------------------------------------------------------------
beta, gamma, delta = 1.0, 0.8, 1.2
kappa1 = 0.05            # small killing
alpha0 = 1.0
# sigmoid feedback varsigma = tanh; gain g chosen below to produce bistability.

def varsigma(x):
    return np.tanh(x)

def dvarsigma(x):
    return 1.0 / np.cosh(x) ** 2     # sech^2

def Qmat(alpha):
    """The 3x3 Metzler sub-generator at activation rate alpha."""
    return np.array([
        [-(alpha + kappa1), alpha, 0.0],
        [beta, -(beta + gamma), gamma],
        [delta, 0.0, -delta],
    ])

dQ_dalpha = np.array([[-1.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])   # dQ/dalpha (only state-1 row)

def principal_data(Q):
    """Lambda>0, nu (left Perron, prob row), h (right, nu.h=1), group inverse R."""
    n = Q.shape[0]
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
    A = Q + Lam * np.eye(n)
    P = np.outer(h, nu)
    R = np.linalg.solve(A + P, np.eye(n) - P)   # group inverse, paper's formula
    return Lam, nu, h, R

def alpha_response(Q):
    """Return (nu, Lambda, dnu/dalpha, dLambda/dalpha) via paper formulas (4),(5)."""
    n = Q.shape[0]
    one = np.ones(n)
    Lam, nu, h, R = principal_data(Q)
    # (4) Hellmann-Feynman ; (5) reduced-resolvent, with simplex correction c
    dLam = -nu @ dQ_dalpha @ h
    c = nu @ dQ_dalpha @ R @ one
    dnu = -nu @ dQ_dalpha @ R + c * nu
    return nu, Lam, dnu, dLam

# ----------------------------------------------------------------------
# G(w) := nu_1 at alpha = alpha0 exp(g varsigma(w)); the self-consistency curve is
# parametric: u = G(w), theta = G(w) - w   (so w = u - theta).  Fold <=> G'(w)=1.
# ----------------------------------------------------------------------
def make_G(g):
    def G(w):
        nu, _, _, _ = alpha_response(Qmat(alpha0 * np.exp(g * varsigma(w))))
        return nu[0]
    return G

def Gprime(g, w):
    """G'(w) analytically: (dnu1/dalpha) * dalpha/dw, dalpha/dw = alpha g varsigma'."""
    alpha = alpha0 * np.exp(g * varsigma(w))
    nu, Lam, dnu, dLam = alpha_response(Qmat(alpha))
    dalpha_dw = alpha * g * dvarsigma(w)
    return dnu[0] * dalpha_dw

# ----------------------------------------------------------------------
# STEP (a): fix the gain g = -5 (the value reported in the paper) and
# confirm bistability (max_w G'(w) > 1, i.e. two folds).
# ----------------------------------------------------------------------
wgrid = np.linspace(-3.0, 3.0, 4001)
g = -5.0
Gp = np.array([Gprime(g, w) for w in wgrid])
maxGp = Gp.max()
print("=== chosen parameters ===")
print(f"beta={beta}, gamma={gamma}, delta={delta}, kappa1={kappa1}, alpha0={alpha0}, g={g}")
print(f"max_w G'(w) = {maxGp:.4f}  (>1 => bistable S-curve)")
if maxGp <= 1.0:
    print("NOT bistable at this gain; STOP (no figure).")
    sys.exit(1)

# fold locations: G'(w) = 1
sign = np.sign(Gp - 1.0)
folds = []
for i in range(len(wgrid) - 1):
    if sign[i] != sign[i + 1] and sign[i] != 0:
        # bisection refine
        a, b = wgrid[i], wgrid[i + 1]
        for _ in range(80):
            m = 0.5 * (a + b)
            if np.sign(Gprime(g, m) - 1.0) == np.sign(Gprime(g, a) - 1.0):
                a = m
            else:
                b = m
        folds.append(0.5 * (a + b))
print(f"fold(s) at w = {['%.5f' % f for f in folds]}  (G'(w)=1)")
if len(folds) < 2:
    print("Fewer than two folds => not bistable; STOP (no figure).")
    sys.exit(1)

# S-curve (parametric) for plotting and to confirm non-monotone theta(w)
u_curve = np.array([make_G(g)(w) for w in wgrid])
theta_curve = u_curve - wgrid
print(f"theta(w) non-monotone: {np.any(np.diff(theta_curve) > 0) and np.any(np.diff(theta_curve) < 0)}")

# ----------------------------------------------------------------------
# STEP (b)+(c): approach one fold; compute responses; fit exponents.
# ----------------------------------------------------------------------
w_f = folds[0]
# approach from the stable side (|G'-1| decreasing to 0); use geometric spacing
deltas = np.geomspace(1e-1, 1e-4, 25)
# move along w toward w_f from one side; pick side where we stay on a real branch
side = -1.0 if w_f - 0.2 > wgrid[0] else +1.0
ws = w_f + side * deltas

sigmas, ndPhi, ngLam, lamG = [], [], [], []
one = np.ones(3)
for w in ws:
    alpha = alpha0 * np.exp(g * varsigma(w))
    Q = Qmat(alpha)
    nu, Lam, dnu_da, dLam_da = alpha_response(Q)
    dalpha_dth = -g * dvarsigma(w) * alpha      # d alpha/d theta
    dalpha_du = +g * dvarsigma(w) * alpha       # d alpha/d u
    dnu_dth = dnu_da * dalpha_dth               # partial_theta nu (vector)
    dnu_du = dnu_da * dalpha_du                 # partial_u nu (vector)
    dLam_dth = dLam_da * dalpha_dth
    dLam_du = dLam_da * dalpha_du
    dF_dth = dnu_dth[0]                          # F = nu_1 - u
    dF_du = dnu_du[0] - 1.0                      # = G'(w) - 1 = sigma~ (signed)
    sigma = abs(dF_du)
    du_star_dth = -dF_dth / dF_du               # closed-loop sensitivity
    dPhi_star = dnu_dth + dnu_du * du_star_dth   # (11) closed-loop QSD Jacobian (vec)
    gLam_star = dLam_dth + dLam_du * du_star_dth  # closed-loop decay gradient (scalar)
    Gstar = float(np.sum(dPhi_star ** 2 / nu))   # single-parameter Fisher = lambda_max
    sigmas.append(sigma)
    ndPhi.append(np.linalg.norm(dPhi_star))
    ngLam.append(abs(gLam_star))
    lamG.append(Gstar)

sigmas = np.array(sigmas)
ndPhi = np.array(ndPhi)
ngLam = np.array(ngLam)
lamG = np.array(lamG)

print(f"\nsigma fit range: [{sigmas.min():.3e}, {sigmas.max():.3e}]  "
      f"= {np.log10(sigmas.max() / sigmas.min()):.2f} decades")

def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]

s_dPhi = slope(sigmas, ndPhi)
s_gLam = slope(sigmas, ngLam)
s_G = slope(sigmas, lamG)
print("\n=== fitted log-log exponents (norm ~ sigma^slope) approaching fold ===")
print(f"  ||dPhi*||     slope = {s_dPhi:+.4f}   (predicted -1)")
print(f"  |grad Lam*|   slope = {s_gLam:+.4f}   (predicted -1)")
print(f"  lambda_max(G*) slope = {s_G:+.4f}   (predicted -2)")

TOL = 0.12
ok = (abs(s_dPhi + 1) < TOL) and (abs(s_gLam + 1) < TOL) and (abs(s_G + 2) < TOL)
print(f"\nexponents match (-1,-1,-2) within {TOL}: {ok}")

if not ok:
    print("GUARDRAIL: exponents do NOT match; STOPPING WITHOUT producing a figure.")
    sys.exit(2)

# ----------------------------------------------------------------------
# STEP (d): figure (vector PDF) -- only reached if exponents matched.
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))

# (a) S-curve: u* vs theta, folds marked
ax1.plot(theta_curve, u_curve, "-", color="#1f3b73", lw=1.8)
for wf in folds:
    uf, thf = make_G(g)(wf), make_G(g)(wf) - wf
    ax1.plot(thf, uf, "o", color="#c1121f", ms=6, zorder=5)
ax1.annotate("Fold", xy=(make_G(g)(folds[0]) - folds[0], make_G(g)(folds[0])),
             xytext=(8, -14), textcoords="offset points", color="#c1121f", fontsize=11)
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$u^{*}=\nu_1$")
ax1.set_title(r"(a) Self-consistency curve $F(\theta,u)=\nu_1-u=0$")
ax1.grid(alpha=0.25)

# (b) log-log lambda_max(G*) vs sigma, slope ~ -2
ax2.loglog(sigmas, lamG, "o", color="#1f3b73", ms=4, label=r"$\lambda_{\max}(\mathcal{G}^{*})$")
c = np.exp(np.log(lamG).mean() - s_G * np.log(sigmas).mean())
ax2.loglog(sigmas, c * sigmas ** s_G, "-", color="#c1121f", lw=1.5,
           label=fr"Fitted slope $={s_G:.2f}$")
ax2.loglog(sigmas, lamG[0] * (sigmas / sigmas[0]) ** (-2.0), "--", color="gray", lw=1.0,
           label=r"Slope $-2$ (predicted)")
ax2.set_xlabel(r"$\sigma=|\partial_u F|$")
ax2.set_ylabel(r"$\lambda_{\max}(\mathcal{G}^{*})$")
ax2.set_title(r"(b) Rank-one fold divergence, $\mathcal{G}^{*}\sim\sigma^{-2}$")
ax2.legend(frameon=False)
ax2.grid(alpha=0.25, which="both")

fig.tight_layout()
out = "fold_example.pdf"
fig.savefig(out)
print(f"\nFIGURE WRITTEN: {out}")

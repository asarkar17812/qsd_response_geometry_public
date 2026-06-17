"""
Numerical verification of the core identities for the QSD response-geometry paper.
Every boxed/displayed formula in the paper is checked here by finite differences
against a parametrized irreducible Metzler sub-generator family Q(theta).
"""
import numpy as np
rng = np.random.default_rng(0)
np.set_printoptions(precision=6, suppress=True)

n = 4  # state-space size of the living class T

def make_Q(theta):
    """A smooth family of irreducible Metzler sub-generators on n states.
    theta has length p; we build off-diagonals as positive exp(...) so the
    matrix is genuinely Metzler and irreducible, with a killing vector q>=0,
    q!=0 (killing only out of state 0, say)."""
    p = len(theta)
    # base off-diagonal log-rates (fixed), perturbed linearly by theta
    A = rng.normal(size=(n, n))
    # deterministic base so the family is reproducible across calls:
    base = np.array([[0.0, 0.3, -0.2, 0.1],
                     [0.2, 0.0, 0.4, -0.1],
                     [-0.3, 0.1, 0.0, 0.5],
                     [0.25, -0.2, 0.15, 0.0]])
    # parameter coupling matrices C_j (one per parameter), fixed
    Clist = []
    for j in range(p):
        Cj = np.zeros((n, n))
        # couple parameter j to a couple of off-diagonal log-rates
        Cj[(j) % n, (j + 1) % n] = 1.0
        Cj[(j + 2) % n, (j + 3) % n] = 0.5
        Clist.append(Cj)
    L = base.copy()
    for j in range(p):
        L = L + theta[j] * Clist[j]
    Off = np.exp(L)              # strictly positive off-diagonals -> irreducible
    np.fill_diagonal(Off, 0.0)
    Q = Off.copy()
    # killing vector: positive killing only out of state 0 (mimics AC->D)
    kill = np.zeros(n)
    kill[0] = 0.7 + 0.1 * np.sum(theta)   # depends smoothly on theta, stays >0 for our theta
    rowsum_off = Off.sum(axis=1)
    diag = -(rowsum_off + kill)
    np.fill_diagonal(Q, diag)
    return Q  # Q1 = -kill <= 0, kill[0]>0 so proper killing

def principal_data(Q):
    """Return Lambda>0, nu (left, prob row), h (right, nu.h=1), and group inverse R."""
    n = Q.shape[0]
    w, vr = np.linalg.eig(Q)            # right eigenvectors
    # dominant = largest real part
    k = int(np.argmax(w.real))
    lam = w[k].real                     # = -Lambda
    Lam = -lam
    h = vr[:, k].real
    # left eigenvector
    wl, vl = np.linalg.eig(Q.T)
    kl = int(np.argmax(wl.real))
    nu = vl[:, kl].real
    # normalize: nu is a probability row vector, nu.h = 1
    if nu.sum() < 0: nu = -nu
    nu = nu / nu.sum()
    if h @ nu < 0: h = -h
    h = h / (nu @ h)
    # group inverse R = (Q + Lam I)^# : index-1, kernel span(h), coker span(nu)
    A = Q + Lam * np.eye(n)
    P = np.outer(h, nu)                 # spectral projection, P=h nu, P^2=P
    # R = (A + P)^{-1} (I - P)   is the group inverse for an index-1 A
    R = np.linalg.solve(A + P, np.eye(n) - P)
    return Lam, nu, h, R, P

def grad_fd(f, theta, eps=1e-6):
    """Central finite-difference gradient/Jacobian of vector- or scalar-valued f."""
    p = len(theta)
    f0 = np.atleast_1d(f(theta))
    J = np.zeros((len(f0), p))
    for j in range(p):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        J[:, j] = (np.atleast_1d(f(tp)) - np.atleast_1d(f(tm))) / (2 * eps)
    return J if len(f0) > 1 else J[0]

# ---- pick an operating point ----
p = 3
theta0 = np.array([0.10, -0.05, 0.08])
Q0 = make_Q(theta0)
Lam0, nu0, h0, R0, P0 = principal_data(Q0)
one = np.ones(n)

print("=== sanity: Metzler, killing, eigenpair ===")
Off = Q0 - np.diag(np.diag(Q0))
print("off-diagonals all >= 0:", np.all(Off >= -1e-14))
print("Q1 (= -kill) <= 0   :", Q0 @ one)
print("Lambda > 0          :", Lam0)
print("nu>0, h>0           :", np.all(nu0 > 0), np.all(h0 > 0))
print("nu Q + Lam nu  ~ 0  :", np.max(np.abs(nu0 @ Q0 + Lam0 * nu0)))
print("Q h + Lam h    ~ 0  :", np.max(np.abs(Q0 @ h0 + Lam0 * h0)))
print("nu.1=1, nu.h=1      :", nu0 @ one, nu0 @ h0)
print("Lambda = nu.kill    :", Lam0, nu0 @ (-(Q0 @ one)))

print("\n=== group inverse axioms ===")
A0 = Q0 + Lam0 * np.eye(n)
print("A R = I - P  ~ 0    :", np.max(np.abs(A0 @ R0 - (np.eye(n) - P0))))
print("R A = I - P  ~ 0    :", np.max(np.abs(R0 @ A0 - (np.eye(n) - P0))))
print("R h ~ 0             :", np.max(np.abs(R0 @ h0)))
print("nu R ~ 0            :", np.max(np.abs(nu0 @ R0)))
print("R P ~ 0, P R ~ 0    :", np.max(np.abs(R0 @ P0)), np.max(np.abs(P0 @ R0)))

print("\n=== (1) Hellmann-Feynman: dLambda/dtheta_j = - nu (dQ/dtheta_j) h ===")
dLam_fd = grad_fd(lambda t: principal_data(make_Q(t))[0], theta0)
# analytic
dQ = grad_fd(lambda t: make_Q(t).reshape(-1), theta0).reshape(n, n, p)
dLam_an = np.array([-nu0 @ dQ[:, :, j] @ h0 for j in range(p)])
print("FD       :", dLam_fd)
print("analytic :", dLam_an)
print("max abs err:", np.max(np.abs(dLam_fd - dLam_an)))

print("\n=== (2) QSD response: dnu/dtheta_j = -nu (dQ_j) R + (nu (dQ_j) R 1) nu ===")
def nu_of(t):
    return principal_data(make_Q(t))[1]
Jnu_fd = grad_fd(nu_of, theta0)             # n x p
Jnu_an = np.zeros((n, p))
for j in range(p):
    cj = nu0 @ dQ[:, :, j] @ R0 @ one
    Jnu_an[:, j] = -nu0 @ dQ[:, :, j] @ R0 + cj * nu0
print("max abs err dnu:", np.max(np.abs(Jnu_fd - Jnu_an)))
print("columns sum to 0 (simplex tangency):", np.max(np.abs(Jnu_an.sum(axis=0))))

print("\n=== (3) projection derivative: dP/dtheta_j = -(R dQ_j P + P dQ_j R) ===")
def P_of(t):
    return principal_data(make_Q(t))[4].reshape(-1)
JP_fd = grad_fd(P_of, theta0).reshape(n, n, p)
err = 0.0
for j in range(p):
    JP_an = -(R0 @ dQ[:, :, j] @ P0 + P0 @ dQ[:, :, j] @ R0)
    err = max(err, np.max(np.abs(JP_fd[:, :, j] - JP_an)))
print("max abs err dP:", err)

print("\n=== (3b) right-eigenvector response: dh/dtheta_j = -R (dQ_j) h + d_j h, d_j = -c_j ===")
# Paper eq. (6); the dual of (2) under the normalization nu.h = 1, so that
# nu.(dh) = d_j = -c_j with c_j = nu (dQ_j) R 1.  Included so the script also covers the
# right-eigenvector response (Remark 4.8).
def h_of(t):
    return principal_data(make_Q(t))[2]
Jh_fd = grad_fd(h_of, theta0)               # n x p, h normalized by nu.h = 1 in principal_data
Jh_an = np.zeros((n, p))
for j in range(p):
    cj = nu0 @ dQ[:, :, j] @ R0 @ one
    Jh_an[:, j] = -R0 @ dQ[:, :, j] @ h0 - cj * h0      # d_j = -c_j
print("max abs err dh:", np.max(np.abs(Jh_fd - Jh_an)))
print("nu.(dh) = -c_j check:", np.max(np.abs(np.array([nu0 @ Jh_an[:, j] + (nu0 @ dQ[:, :, j] @ R0 @ one) for j in range(p)]))))

print("\n=== (4) Fisher info = KL Hessian = J^T diag(1/nu) J, ker(Fisher)=ker(J) ===")
G_fisher = Jnu_an.T @ np.diag(1.0 / nu0) @ Jnu_an
def KL(a, b):
    return float(np.sum(a * np.log(a / b)))
# Hessian in theta' of KL(nu(theta') || nu(theta0)) at theta'=theta0
H = np.zeros((p, p))
eps = 1e-4
for i in range(p):
    for j in range(p):
        tpp = theta0.copy(); tpp[i] += eps; tpp[j] += eps
        tpm = theta0.copy(); tpm[i] += eps; tpm[j] -= eps
        tmp = theta0.copy(); tmp[i] -= eps; tmp[j] += eps
        tmm = theta0.copy(); tmm[i] -= eps; tmm[j] -= eps
        H[i, j] = (KL(nu_of(tpp), nu0) - KL(nu_of(tpm), nu0)
                   - KL(nu_of(tmp), nu0) + KL(nu_of(tmm), nu0)) / (4 * eps**2)
print("Fisher (J^T D^-1 J):\n", G_fisher)
print("KL Hessian (FD):\n", H)
print("max abs err Fisher vs KL-Hessian:", np.max(np.abs(G_fisher - H)))

print("\n=== (5) scaling gauge: Q -> s Q gives nu fixed, Lambda -> s Lambda ===")
s = 1.7
Lam_s, nu_s, h_s, _, _ = principal_data(s * Q0)
print("nu unchanged    :", np.max(np.abs(nu_s - nu0)))
print("Lambda scales s :", Lam_s, s * Lam0, abs(Lam_s - s * Lam0))
print("=> ker(J_nu) contains scaling dir, but dLambda/dlog s = Lambda != 0 (channel split)")

print("\n=== (6) conservative (no-killing) limit: R -> Q^# = -Z, dnu -> -pi (dQ) Q^# ===")
# Build a conservative generator (no killing): set kill=0 by using a separate builder
def make_Qcons(theta):
    base = np.array([[0.0, 0.3, -0.2, 0.1],
                     [0.2, 0.0, 0.4, -0.1],
                     [-0.3, 0.1, 0.0, 0.5],
                     [0.25, -0.2, 0.15, 0.0]])
    L = base.copy()
    for j in range(len(theta)):
        Cj = np.zeros((n, n))
        Cj[(j) % n, (j + 1) % n] = 1.0
        Cj[(j + 2) % n, (j + 3) % n] = 0.5
        L = L + theta[j] * Cj
    Off = np.exp(L); np.fill_diagonal(Off, 0.0)
    Q = Off.copy()
    np.fill_diagonal(Q, -Off.sum(axis=1))   # conservative: row sums 0
    return Q
Qc = make_Qcons(theta0)
Lamc, pic, hc, Rc, Pc = principal_data(Qc)
print("Lambda ~ 0 (conservative):", Lamc)
print("h ~ 1 (const right evec) :", np.max(np.abs(hc - one)))
# stationary distribution pi solves pi Q = 0
print("pi Q ~ 0                 :", np.max(np.abs(pic @ Qc)))
# fundamental matrix Z = (1 pi - Q)^{-1} - 1 pi ; claim Q^# = -Z
Z = np.linalg.inv(np.outer(one, pic) - Qc) - np.outer(one, pic)
Qsharp = Rc   # group inverse of (Q + 0) = Q^#
print("Q^# = -Z  ~ 0            :", np.max(np.abs(Qsharp + Z)))
# stationary sensitivity: dpi/dtheta = -pi (dQ) Q^# = +pi (dQ) Z
dQc = grad_fd(lambda t: make_Qcons(t).reshape(-1), theta0).reshape(n, n, p)
Jpi_fd = grad_fd(lambda t: principal_data(make_Qcons(t))[1], theta0)
Jpi_an = np.zeros((n, p))
for j in range(p):
    Jpi_an[:, j] = -pic @ dQc[:, :, j] @ Qsharp   # = +pi (dQ) Z
print("max err dpi vs -pi(dQ)Q^#:", np.max(np.abs(Jpi_fd - Jpi_an)))

print("\n=== (7) sign check on the canonical 2-state chain (paper's check) ===")
# 1->2 at rate a, 2->1 at rate b, conservative; pi_1 = b/(a+b)
a, b = 1.0, 1.0
def Q2(ab):
    a, b = ab
    return np.array([[-a, a], [b, -b]])
Lam2, pi2, h2, R2, P2 = principal_data(Q2([a, b]))
dpi1_da_fd = (principal_data(Q2([a + 1e-6, b]))[1][0] - principal_data(Q2([a - 1e-6, b]))[1][0]) / 2e-6
print("d pi_1/d a  exact -b/(a+b)^2 =", -b / (a + b)**2, " FD =", dpi1_da_fd)
dQ2_da = np.array([[-1.0, 1.0], [0.0, 0.0]])
Qsharp2 = R2
print("-pi(dQ)Q^# gives           :", (-pi2 @ dQ2_da @ Qsharp2)[0], " (should match -1/4)")
print("\nALL CHECKS COMPLETE")

print("\n=== (8) second-order eigenvalue (curvature of Lambda) ===")
# For simple eigenvalue lam_* = -Lambda of Q, reduced resolvent R=(Q+Lam I)^#:
#   d_i d_j lam_* = nu (d_ij Q) h  - nu (d_i Q) R (d_j Q) h - nu (d_j Q) R (d_i Q) h
# hence Hessian of Lambda is the negative of that.
def lam_star(t):
    Q = make_Q(t); w = np.linalg.eigvals(Q)
    return float(w[np.argmax(w.real)].real)
# FD Hessian of lam_star
Hl = np.zeros((p, p)); e = 1e-4
for i in range(p):
    for j in range(p):
        tpp=theta0.copy(); tpp[i]+=e; tpp[j]+=e
        tpm=theta0.copy(); tpm[i]+=e; tpm[j]-=e
        tmp=theta0.copy(); tmp[i]-=e; tmp[j]+=e
        tmm=theta0.copy(); tmm[i]-=e; tmm[j]-=e
        Hl[i,j]=(lam_star(tpp)-lam_star(tpm)-lam_star(tmp)+lam_star(tmm))/(4*e*e)
# analytic; need d_ij Q. Our Q depends linearly on theta in L but exp() and kill add curvature.
def Qflat(t): return make_Q(t).reshape(-1)
def d2Q(i,j):
    tpp=theta0.copy(); tpp[i]+=e; tpp[j]+=e
    tpm=theta0.copy(); tpm[i]+=e; tpm[j]-=e
    tmp=theta0.copy(); tmp[i]-=e; tmp[j]+=e
    tmm=theta0.copy(); tmm[i]-=e; tmm[j]-=e
    return ((make_Q(tpp)-make_Q(tpm)-make_Q(tmp)+make_Q(tmm))/(4*e*e))
Hl_an=np.zeros((p,p))
for i in range(p):
    for j in range(p):
        Hl_an[i,j]= nu0@d2Q(i,j)@h0 - nu0@dQ[:,:,i]@R0@dQ[:,:,j]@h0 - nu0@dQ[:,:,j]@R0@dQ[:,:,i]@h0
print("max err d2 lam_star:", np.max(np.abs(Hl-Hl_an)))

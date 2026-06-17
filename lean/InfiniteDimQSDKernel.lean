/-
  Machine-checked algebraic kernel of the infinite-dimensional QSD response geometry (Section 8).

  The infinite-dimensional theorems (QSD existence, the group-inverse response calculus, the fold
  amplification, the Λ→0 recovery) reduce, on every finite truncation, to exact algebraic identities
  about the principal eigentriple (Λ, ν, h) of a killed sub-generator Q and its group inverse
  R = (Q + Λ I)^#.  This file verifies that kernel, EXACTLY over the rationals, on the canonical
  2-state instance also used in code/infinite_dim_qsd_kernel.py:

      Q = [[-2, 1], [1, -2]]   (a Metzler sub-generator: off-diagonals ≥ 0, row sums = -1 < 0,
                                so each state is killed at rate 1),
      principal eigenvalue  -Λ = -1,  right eigenvector h = (1,1),  left eigenvector ν = (1/2,1/2),
      group inverse  R = [[-1/4, 1/4], [1/4, -1/4]].

  Verified (all by `native_decide` over ℚ -- no Mathlib, Lean-core `Rat` only):
    (E)  Q h = -Λ h   and   νᵀ Q = -Λ νᵀ      (eigen-relations)
    (N)  ν · h = 1                            (normalization)
    (K1) (Q + Λ I) R = I - h⊗ν                (the group-inverse identity)
    (K1') R (h⊗ν) = 0  and  (h⊗ν) R = 0       (R annihilates the principal projection)

  This is the machine-checkable base case of the limit argument; the analytic passage to the
  infinite-dimensional limit (Lyapunov drift, Krein-Rutman, Kato) is classical and cited in the paper.
-/

namespace InfiniteDimQSDKernel

-- Q + Λ I  (Λ = 1), entries aᵢⱼ
def a00 : Rat := -1
def a01 : Rat := 1
def a10 : Rat := 1
def a11 : Rat := -1

-- group inverse R
def r00 : Rat := -1/4
def r01 : Rat := 1/4
def r10 : Rat := 1/4
def r11 : Rat := -1/4

-- (A R)ᵢⱼ  with A = Q + Λ I
def AR00 : Rat := a00 * r00 + a01 * r10
def AR01 : Rat := a00 * r01 + a01 * r11
def AR10 : Rat := a10 * r00 + a11 * r10
def AR11 : Rat := a10 * r01 + a11 * r11

-- Π = h ⊗ ν, with h = (1,1), ν = (1/2,1/2)  ⇒  Πᵢⱼ = hᵢ νⱼ = 1/2
-- (I - Π)ᵢⱼ
def ImPi00 : Rat := 1 - 1/2
def ImPi01 : Rat := 0 - 1/2
def ImPi10 : Rat := 0 - 1/2
def ImPi11 : Rat := 1 - 1/2

/-- (K1) The group-inverse identity (Q + Λ I) R = I - h⊗ν, exactly over ℚ. -/
theorem groupinverse_identity :
    AR00 = ImPi00 ∧ AR01 = ImPi01 ∧ AR10 = ImPi10 ∧ AR11 = ImPi11 := by
  native_decide

-- Eigen-relations.  Q entries (qᵢⱼ), h = (1,1), ν = (1/2,1/2), Λ = 1.
def q00 : Rat := -2
def q01 : Rat := 1
def q10 : Rat := 1
def q11 : Rat := -2

-- (Q h)ᵢ  vs  -Λ hᵢ = -1
theorem Qh_eq_negLam_h :
    (q00 * 1 + q01 * 1 = (-1 : Rat)) ∧ (q10 * 1 + q11 * 1 = (-1 : Rat)) := by
  native_decide

-- (νᵀ Q)ⱼ  vs  -Λ νⱼ = -1/2
theorem nuQ_eq_negLam_nu :
    ((1/2 : Rat) * q00 + (1/2) * q10 = -1/2) ∧ ((1/2 : Rat) * q01 + (1/2) * q11 = -1/2) := by
  native_decide

/-- (N) Normalization ν · h = 1. -/
theorem nu_dot_h : (1/2 : Rat) * 1 + (1/2) * 1 = 1 := by native_decide

-- (K1') R annihilates Π = h⊗ν on both sides: (R Π)ᵢⱼ = 0 and (Π R)ᵢⱼ = 0.
-- Πᵢⱼ = 1/2 for all i,j.
def RPi00 : Rat := r00 * (1/2) + r01 * (1/2)
def RPi10 : Rat := r10 * (1/2) + r11 * (1/2)
def PiR00 : Rat := (1/2) * r00 + (1/2) * r10
def PiR01 : Rat := (1/2) * r01 + (1/2) * r11

theorem R_annihilates_Pi :
    RPi00 = 0 ∧ RPi10 = 0 ∧ PiR00 = 0 ∧ PiR01 = 0 := by
  native_decide

end InfiniteDimQSDKernel

/-
  Machine-checked rational core of Example 8.6: the countable birth-death-with-killing chain on a
  non-compact state space, its explicit Foster-Lyapunov drift, and its exactly-solvable Poisson QSD.
  This supplies the drift function that the infinite-dimensional existence result reduces to on the
  unbounded active count n ∈ {0,1,2,...}:

      birth   n → n+1   at rate  β            (constant)
      death   n → n-1   at rate  μ·n
      killing n → †      at rate  κ·n          (absorbing)

  with the calibrated rationals  β = 1, μ = 1/2, κ = 1/10.  Two algebraic facts, both over ℚ:

  (A) FOSTER-LYAPUNOV DRIFT.  For W(n) = e^{γ n}, (𝓛W/W)(n) = β(e^γ-1) - n[μ(1-e^{-γ})+κ] is AFFINE
      in n with negative slope.  Using the rational exponential bounds valid at γ=1,
          e^γ - 1 = e - 1 ≤ 7/4      (since e ≤ 11/4),
          1 - e^{-γ} = 1 - 1/e ≥ 5/8 (since 1/e ≤ 3/8),
      the drift upper bound  Dbar(n) = β·(7/4) - n·(μ·(5/8)+κ)  has per-step slope 33/80 > 0 and
      Dbar(7) = -91/80 ≤ -1 = -c, so 𝓛W ≤ -c W + b·1_K with K = {0,...,6}, N0 = 7.  (The two
      transcendental bounds e ≤ 11/4, 1/e ≤ 3/8 are true and checked numerically in the analysis
      module; here they are the explicit rational hypotheses the drift arithmetic consumes.)

  (B) EXACT QSD (the chain is exactly solvable).  The QSD is Poisson(ρ), ρ = β/(μ+κ), with
      principal eigenvalue Λ = κβ/(μ+κ) and gap μ+κ.  Poisson stationarity reduces to the per-state
      eigen-relation (𝓛*ν)(m)/ν(m) = m·[β/ρ - (μ+κ)] + [μρ - β]: the m-coefficient VANISHES and the
      constant equals -Λ.  Both verified exactly over ℚ (ρ = 5/3, Λ = 1/6, gap = 3/5).

  All by `native_decide` over ℚ (Lean-core `Rat`, no Mathlib).  The analytic content -- that drift +
  the finite small set + irreducibility yield a unique QSD with a gap on the non-compact chain
  (Champagnat-Villemonais; Ferré-Rousset) -- is beyond Lean-core and is classical and cited in the paper.
-/

namespace UnboundedLyapunovKernel

-- calibrated rational rates
def beta  : Rat := 1
def mu    : Rat := 1/2
def kappa : Rat := 1/10
def c     : Rat := 1          -- drift margin

-- rational exponential bounds at γ = 1 (numerically: e-1 ≈ 1.71828 ≤ 7/4; 1-1/e ≈ 0.63212 ≥ 5/8)
def a_up : Rat := 7/4         -- ≥ e^γ - 1
def d_lo : Rat := 5/8         -- ≤ 1 - e^{-γ}

/-- drift upper bound  Dbar(n) = β·a_up - n·(μ·d_lo + κ),  affine in n. -/
def Dbar (n : Rat) : Rat := beta * a_up - n * (mu * d_lo + kappa)

-- (A1) the per-step slope μ·d_lo + κ = 33/80 is strictly positive (Dbar decreasing).
theorem drift_slope_pos : mu * d_lo + kappa = 33/80 ∧ (mu * d_lo + kappa > 0) := by native_decide

-- (A2) at the threshold N0 = 7 the drift bound is ≤ -c (and equals -91/80).
theorem drift_at_N0 : Dbar 7 = -91/80 ∧ Dbar 7 ≤ -c := by native_decide

-- (A3) N0 = 7 is the smallest integer crossing: Dbar(6) > -c ≥ Dbar(7) (so K = {0,...,6}).
theorem drift_threshold_sharp : Dbar 6 > -c ∧ Dbar 7 ≤ -c := by native_decide

-- the rational bounds point the right way (e-1 ≤ 7/4 and 1-1/e ≥ 5/8 are the consumed hypotheses);
-- recorded as the exact rationals the affine drift needs.
theorem bounds_consistent : a_up = 7/4 ∧ d_lo = 5/8 := by native_decide

-- (B) exact QSD of the solvable chain
def rho : Rat := beta / (mu + kappa)        -- = 5/3
def Lam : Rat := kappa * beta / (mu + kappa) -- = 1/6
def gap : Rat := mu + kappa                  -- = 3/5

/-- (B1) ρ = 5/3, Λ = 1/6, gap = 3/5 (the closed forms). -/
theorem closed_forms : rho = 5/3 ∧ Lam = 1/6 ∧ gap = 3/5 := by native_decide

/-- (B2) Poisson(ρ) is the QSD: the m-coefficient of the per-state eigen-relation vanishes,
    β/ρ - (μ+κ) = 0  (this is exactly why Poisson is stationary). -/
theorem eigen_m_coefficient_zero : beta / rho - (mu + kappa) = 0 := by native_decide

/-- (B3) the constant term of the eigen-relation equals -Λ:  μ·ρ - β = -Λ. -/
theorem eigen_constant_is_negLam : mu * rho - beta = -Lam := by native_decide

/-- (B4) the QSD identity Λ = κ·E_ν[N] with E_ν[N] = ρ (Poisson mean): κ·ρ = Λ. -/
theorem qsd_identity_Lambda_eq_kappa_EN : kappa * rho = Lam := by native_decide

end UnboundedLyapunovKernel

# Lean kernel scope

Each kernel checks a finite, decidable, rational witness over the rationals: the paper's claimed
identity or constant satisfies its defining relation exactly. This is not a proof of existence,
uniqueness, or the analytic estimate; those are classical and cited in the paper.

| Kernel | Rational core verified (over the rationals) | Cited, not formalized |
|---|---|---|
| `InfiniteDimQSDKernel` | the group-inverse identity `(Q + Lambda I) R = I - h (x) nu`, the eigen-relations, and `nu R = 0` exactly on the 2-state killed instance | Krein-Rutman / Kato on the weighted space; the infinite-dimensional limit |
| `UnboundedLyapunovKernel` | Example 8.6 (Poisson birth-death-with-killing): `rho = 5/3`, `Lambda = kappa rho = 1/6`, gap `mu + kappa = 3/5`, drift `Dbar(7) = -91/80 <= -1`, and the two rational eigenvalue quadratics | Champagnat-Villemonais QSD existence and gap on the non-compact countable chain |

## What no kernel claims

No kernel formalizes Perron-Frobenius, Krein-Rutman, Kato analytic perturbation, or
Champagnat-Villemonais. Those are not in Mathlib to the required depth; they are cited, and the kernels
verify only the rational arithmetic the paper reduces to on concrete instances. See `AXIOMS.md` for the
axiom profile.

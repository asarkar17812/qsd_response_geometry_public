# Quasi-Stationary Response Geometry

Source, figures, verification scripts, a walkthrough notebook, and Lean 4 kernels for the paper

> **Quasi-Stationary Response Geometry: The Perturbation Calculus and Information Geometry of
> Parametrized Sub-Markovian Generators**, A. Sarkar.

The paper studies the principal spectral datum `(Lambda, nu)` of a real-analytic family of irreducible
Metzler matrices with nonpositive, not identically vanishing row sums (the generators of killed
continuous-time Markov chains): the decay rate `Lambda > 0` and the quasi-stationary distribution `nu`.
It develops a closed first-order response calculus organized around the rate channel `Lambda`, the
pulled-back Fisher information geometry of the QSD map, and the fold-transport behavior under a
self-consistent coupling. Sections 2-7 are the proved finite-dimensional core; Section 8 records the
operator-theoretic transport to countable and diffusion state spaces under cited hypotheses.

## Layout

```
paper/      LaTeX source, the compiled PDF, the plain-text arXiv abstract, and the four figures
code/       verification and figure scripts (NumPy / Matplotlib)
lean/        Lean 4 kernels that check the rational cores over the rationals
notebook/   an executable walkthrough that restates and re-checks each numbered result
```

Where the paper's code-availability note names a script under `analysis/` or `figures/`, the file
lives under `code/` here (for example `analysis/eflat_nonuniform_qsd.py` is `code/eflat_nonuniform_qsd.py`).

## Building the paper

The source is a single self-contained file with an embedded bibliography; it needs only a standard
TeX Live installation.

```
cd paper
latexmk -pdf qsd_response_geometry.tex
```

## Reproducing the numerics

```
pip install -r requirements.txt
cd code
python verify_identities.py          # first- and second-order response identities vs finite differences
python eflat_nonuniform_qsd.py       # the e-flat non-uniform witness (Example 5.10, Proposition 5.11)
python unbounded_qsd_lyapunov.py     # the birth-death-with-killing instance (Example 8.6; writes Figure 4)
python infinite_dim_qsd_kernel.py    # group-inverse identity on the 2-state and a 5x5 instance
python figures/scalar_fold.py        # Figure 1
python figures/channel_split.py      # Figure 2
python figures/fold_example.py       # Figure 3 (emits only if the fitted exponents match (-1,-1,-2))
```

Each script prints its checks and writes a JSON record (and, where relevant, a figure) under
`code/output/`. `config.py` is a small shim that provides the output-path helper and the pinned 5x5
generator used by `infinite_dim_qsd_kernel.py`, so the folder runs on its own.

To run the notebook headless:

```
pip install nbconvert ipykernel
jupyter nbconvert --to notebook --execute notebook/qsd_response_geometry_walkthrough.ipynb
```

## Lean kernels

The Lean files check only the **rational cores over the rationals** -- the group-inverse identity
`(Q + Lambda I) R = I - h (x) nu` on a finite instance, and the rational drift and eigenvalue
constants of Example 8.6. They do **not** formalize the analytic content (Perron-Frobenius,
Krein-Rutman, Kato perturbation), which is classical and cited in the paper. Each file is a standalone
check:

```
cd lean
lean UnboundedLyapunovKernel.lean
lean InfiniteDimQSDKernel.lean
```

Per-kernel scope is in `lean/MAPPING.md`; the axiom profile is in `lean/AXIOMS.md`.

## License

See `LICENSE` (a license must be chosen before public release).

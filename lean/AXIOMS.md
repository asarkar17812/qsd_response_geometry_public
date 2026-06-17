# Axiom profile

The only axioms used are Lean's standard `propext`, `Classical.choice`, `Quot.sound`, plus -- where
`native_decide` is used -- the compiler-trust axiom it introduces
(`Lean.ofReduceBool`, surfaced per theorem as a `..._native.native_decide.ax_..` constant). No other
axioms. There are no `sorry` / `admit`.

`native_decide` compiles the proposition to native code and trusts the compiler's evaluation, so it
adds the compiler to the trusted base beyond the Lean kernel. It is used because plain `decide` stalls
on rational kernel reduction. The arithmetic facts checked are small and reproducible by hand and in
the analysis scripts, so the residual trust is the compiler's correctness on closed rational
arithmetic.

## Reproduce

```
# per file (each is a standalone Lean-core file, no Mathlib):
lean UnboundedLyapunovKernel.lean
lean InfiniteDimQSDKernel.lean

# axiom audit for any theorem (append to a copy of the file, then run lean):
#   #print axioms <Namespace>.<theorem>
```

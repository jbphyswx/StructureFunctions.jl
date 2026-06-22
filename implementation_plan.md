# StructureFunctions.jl Transverse, Tensor, and KHM Implementation Plan

## Status

Last updated: 2026-06-21

Overall state: implementation in progress.

Use this checklist as the durable progress tracker.

- [x] Phase 0: Confirm current public behavior and record compatibility risks.
- [x] Phase 1: Fix and document longitudinal/transverse scalar semantics.
- [ ] Phase 2: Refactor CPU/GPU projection helpers and single-pass kernels.
- [x] Phase 3: Add explicit component-normalized transverse operators.
- [x] Phase 4: Add tensor structure-function API, starting with orders 2 and 3.
- [x] Phase 5: Add basic KHM exact-law diagnostics.
- [x] Phase 6: Fix operator/result hierarchy and Helmholtz derived quantities.
- [ ] Phase 7: Update documentation and examples.
- [x] Phase 8: Add parity, stability, and regression tests.
- [ ] Phase 9: Run full verification and update this checklist.

Completed in the first implementation pass:

- Added invariant helpers `transverse_norm2` and `transverse_component_norm2`.
- Added explicit basis convention types:
  - `CanonicalTransverseBasis`
  - `ReferenceAxisTransverseBasis`
  - `CoordinateGaugeTransverseBasis`
  - `UserTransverseBasis`
- Added explicit component-normalized operators:
  - `T2ComponentSF`
  - `L1T2ComponentSF`
- Updated `T2SF` and `L1T2SF` to use total invariant transverse energy.
- Updated `S3SF` to the scalar flux combination `du_L * norm(du)^2`, so `S3SF == L3SF + L1T2SF`.
- Added `StructureFunctionTensor` plus serial `calculate_structure_function_tensor` / `calculate_structure_function_tensor!` for tensor orders 2 and 3, including point, shared-position auxiliary, and varying-position auxiliary shapes.
- Added `StructureFunctions.KHM` with binned KHM exact-law diagnostics.
- Updated CPU serial, CPU auxiliary, OhMyThreads, Distributed, and generic GPU global-atomic single-pass bulk rows to compute invariant transverse energy rather than one signed normal component.
- Split structure-function quantity types into pairwise and derived families.
- Added Helmholtz-derived rotational/divergent second-order quantity types and explicit `helmholtz_decompose_2d` API.

Still intentionally unfinished:

- Add explicit basis-aware scalar calculation APIs for signed/component odd transverse diagnostics.
- Make GPU single-pass launch specialization use `Val{D}` instead of runtime `N_dims` branches in the generic global-atomic kernels.
- Validate `UserTransverseBasis` outputs in the safe public path.
- Add threaded/GPU tensor kernels after the serial API is stable.
- Update public documentation and release notes.
- Run and fix the full test suite, including Aqua/JET/GPU tiled parity.

## Summary

The package currently mixes three different transverse concepts:

1. Total perpendicular vector energy.
2. Per-transverse-component isotropic quantities.
3. Signed transverse scalar components chosen by a convention.

These must be separated before the package can be considered mathematically clean for 3D velocity fields or future higher-dimensional use.

The core invariant decomposition for a point pair is:

```text
du = u(x + r) - u(x)
rhat = r / norm(r)
du_L = dot(du, rhat)
du_L_vec = du_L * rhat
du_T_vec = du - du_L_vec
```

Because `du_T_vec` is perpendicular to `rhat`:

```text
norm(du)^2 = du_L^2 + norm(du_T_vec)^2
```

This plan makes that identity the basis of all invariant scalar operators.

References:

- Longitudinal/transverse mixed structure functions: https://arxiv.org/abs/1012.4070
- Second-order structure-function tensor and AGKE: https://arxiv.org/abs/2005.07438
- KHMH equation context and form ambiguity: https://arxiv.org/abs/2403.10457

Terminology:

- KHM: Karman-Howarth-Monin equation family for scale-by-scale turbulence relations.
- KHMH: Karman-Howarth-Monin-Hill form, commonly used for inhomogeneous/non-stationary two-point budget equations.
- AGKE: Anisotropic Generalized Kolmogorov Equations. These are full budget equations for every component of the second-order structure-function tensor, including production, transport, redistribution/pressure-strain, dissipation, and scale/space fluxes.

This plan includes the second-order tensor API and basic KHM exact-law diagnostics. It does not claim a full AGKE/KHMH budget implementation in the first pass.

## Mathematical Semantics To Implement

### Invariant scalar operators

These do not require a transverse basis and should work for any dimension where the input contract permits the calculation.

```text
S2SF    = < norm(du)^2 >
L2SF    = < du_L^2 >
T2SF    = < norm(du_T_vec)^2 >
L3SF    = < du_L^3 >
S3SF    = < du_L * norm(du)^2 >
L1T2SF  = < du_L * norm(du_T_vec)^2 >
```

Important identity:

```text
S2SF = L2SF + T2SF
S3SF = L3SF + L1T2SF
```

### Component-normalized isotropic transverse operators

These assume isotropic sharing of transverse energy among the `D - 1` directions perpendicular to `rhat`.

```text
T2ComponentSF   = T2SF / (D - 1)
L1T2ComponentSF = L1T2SF / (D - 1)
```

These must be separate public operators. Do not silently change `T2SF` to mean a per-component quantity.

### Signed transverse scalar operators

These require a chosen transverse basis vector `e_T1` and should not be part of default bulk/single-pass outputs.

```text
L2T1SF = < du_L^2 * dot(du, e_T1) >
T3SF   = < dot(du, e_T1)^3 >
```

Expected behavior:

- In 2D, the transverse basis is canonical up to sign.
- In 3D and higher, there is no unique signed transverse direction.
- Signed odd transverse quantities are expected to vanish for isotropic statistics.
- In anisotropic data, signed odd transverse quantities are convention-dependent diagnostics, not invariant scalar structure functions.
- They should be opt-in through explicit basis-aware APIs.

### Transverse basis conventions

For `D >= 3`, a signed transverse basis is a gauge choice. The package should not pretend there is a unique physical transverse scalar.

There are two useful convention families, and they must be named differently:

1. Physical-reference convention. This is interpretable when the flow has a meaningful global direction, such as vertical, rotation axis, mean field, or shear direction.

```text
axis = user reference axis, default zhat for D == 3
candidate = axis - dot(axis, rhat) * rhat
if norm(candidate) is too small:
    throw a clear degeneracy error
e_T1 = normalize(candidate)
```

This has a real singular direction when `rhat` is parallel to the reference axis. That singularity is not a bug; it means this reference cannot define a transverse component for that pair. Do not silently patch it with arbitrary replacement axes in the same type.

2. Computational-gauge convention. This generates an orthonormal basis directly from `rhat`, for example with a Hughes-Moller, Frisvad, or Duff-style ONB construction. It can be fast and avoids near-parallel checks, but the selected transverse direction rotates according to an algorithm rather than the physics. It may wash out or distort signed anisotropic statistics. If added, name it explicitly as `CoordinateGaugeTransverseBasis` and document that it is not a physical reference convention.

Also support bring-your-own basis functions for specialized analyses. For example, a user may want vertical velocity with longitude-like transverse directions, wall-normal references, local magnetic-field references, or a paper-specific convention. These should be explicit inputs rather than hidden package defaults.

For a full local transverse basis, construct remaining basis vectors by Gram-Schmidt in the plane perpendicular to `rhat`.

The existing `n_hat` helper should be kept only as compatibility terminology and documented as "the first signed transverse basis vector selected by the convention", not as "the transverse direction".

## Phase 0: Compatibility Audit

- [ ] List all public names currently tied to transverse behavior:
  - `T2SF`
  - `T3SF`
  - `L2T1SF`
  - `L1T2SF`
  - longhand operator names
  - `magnitude_du_transverse` / `mdu_t`
  - `du_transverse` / `du_t`
  - `n_hat`
- [ ] Identify tests that currently encode signed-normal behavior.
- [ ] Decide which changes are breaking and document them in release notes.
- [ ] Add temporary comments in the plan checklist noting any tests intentionally rewritten.

Acceptance criteria:

- The compatibility risks are listed before code behavior changes.
- No public operator changes are made without a matching test and documentation update.

## Phase 1: Projection Helpers

Add allocation-free helpers in `src/HelperFunctions.jl`, preserving the existing naming style where possible.

Required helpers:

```julia
magnitude_δu_longitudinal(du, rhat)  # existing public name; keep it
δu_longitudinal(du, rhat)            # existing public name; keep it
δu_transverse(du, rhat)              # existing public name; make semantics explicit
transverse_norm2(du, rhat)
transverse_component_norm2(du, rhat)
```

Do not introduce `longitudinal_scalar` as a public replacement for `magnitude_δu_longitudinal`. If a shorter ASCII internal alias is useful for GPU or generated code, keep it private and document it as an alias.

Add transverse basis convention types:

```julia
abstract type AbstractTransverseBasisConvention end
struct CanonicalTransverseBasis <: AbstractTransverseBasisConvention end
struct ReferenceAxisTransverseBasis{A} <: AbstractTransverseBasisConvention
    axis::A
end
struct CoordinateGaugeTransverseBasis <: AbstractTransverseBasisConvention end
struct UserTransverseBasis{F} <: AbstractTransverseBasisConvention
    basis_function::F
end
```

Behavior:

- `CanonicalTransverseBasis` is valid for 2D.
- `ReferenceAxisTransverseBasis` is valid for `D >= 3`.
- Basis construction must throw `ArgumentError` if the reference axis is parallel or nearly parallel to `rhat`.
- The near-parallel tolerance must be explicit and documented.
- `CoordinateGaugeTransverseBasis`, if implemented, must use a fast ONB-from-normal construction and must be documented as a computational gauge, not an invariant or physical convention.
- Do not use `CoordinateGaugeTransverseBasis` as the default for signed statistical operators.
- `UserTransverseBasis` calls `basis_function(rhat)` and validates that returned vectors are perpendicular to `rhat`, unit length, and mutually orthogonal when more than one vector is returned. Validation can be disabled only through an explicitly named unsafe/internal path for performance-sensitive expert use.

Acceptance criteria:

- `transverse_norm2(du, rhat)` equals `norm2(transverse_vector(du, rhat))`.
- `norm2(du)` equals `magnitude_δu_longitudinal(du, rhat)^2 + transverse_norm2(du, rhat)` to roundoff.
- Constructed signed basis vectors are unit length and perpendicular to `rhat`.

## Phase 2: Scalar Operator Semantics

Update `src/StructureFunctionTypes.jl`.

Required behavior:

- `ProjectedStructureFunctionType{0,2}` uses total transverse energy.
- `ProjectedStructureFunctionType{1,2}` uses `du_L * transverse_norm2(du, rhat)`.
- `ProjectedStructureFunctionType{2,1}` and `ProjectedStructureFunctionType{0,3}` are basis-dependent signed operators and must not run in `D >= 3` without an explicit transverse basis convention.
- Existing `S3SF` / `ThirdOrderStructureFunction` stays public; do not add a duplicate. Document and test it as `L3SF + L1T2SF`.
- Default single-pass outputs should include invariant bulk operators only.

Add explicit component-normalized types:

```julia
TransverseComponentSecondOrderStructureFunctionType
LongitudinalTransverseComponentThirdOrderStructureFunctionType
```

Add singleton aliases:

```julia
T2ComponentSF
L1T2ComponentSF
```

Resolution API:

- Add these names to `get_structure_function_type`.
- Do not overload `calculate_structure_function` shorthand APIs again.

Acceptance criteria:

- `S2SF == L2SF + T2SF` for exact single-pair tests.
- `S3SF == L3SF + L1T2SF` for exact single-pair tests.
- `T2ComponentSF == T2SF / (D - 1)`.
- `L1T2ComponentSF == L1T2SF / (D - 1)`.

## Phase 3: CPU And GPU Kernel Cleanup

The current single-pass fast paths must stop using a single signed normal component for invariant transverse energy.

CPU:

- Replace local `du_T` scalar usage in invariant paths with:
  - `du_T2 = transverse_norm2(du, rhat)`
  - `du_T_component = transverse_component(du, rhat, basis, basis_index)` only for basis-dependent component operators.
- Remove `L2T1SF` and `T3SF` from single-pass outputs for this pass.
- Default single-pass outputs contain invariant bulk values only, including `S2SF`, `L2SF`, `T2SF`, `S3SF`, `L3SF`, and `L1T2SF`.
- Basis-dependent operators remain available only through explicit basis-aware non-single-pass calculations until a separate design justifies optimized single-pass support.
- Keep 2D performance by specializing through `Val{D}` and inlining.

GPU:

- Replace runtime `N_dims == 3` ternaries in generic global-atomic kernels with `Val{D}` launch specialization.
- Add device helpers for:
  - `du_L`
  - `du_T2_total`
  - signed first transverse scalar for supported conventions.
- For `D == 3`, `T2SF` must include both perpendicular components.
- Do not include signed basis-dependent operators in default GPU single-pass outputs.
- Do not add GPU single-pass support for basis-dependent signed/component operators in this pass.

Acceptance criteria:

- A 3D fixture with nonzero transverse residual in two components fails under the old formula and passes under the new formula.
- CPU serial, CPU threaded, KA.CPU GPU, and CUDA when available agree for supported invariant operators.
- Kernel code no longer branches on runtime dimension for the generic global-atomic path.

## Phase 4: Tensor API, Starting With Orders 2 And 3

Add a tensor API that is separate from scalar/projected operators.

Public API:

```julia
calculate_structure_function_tensor(order::Val, x, u, distance_bins; backend=AutoBackend(), kwargs...)
calculate_structure_function_tensor!(sums, counts, order::Val, x, u, distance_bins; backend=SerialBackend(), kwargs...)
```

Result type:

```julia
StructureFunctionTensor{P}
```

Storage contract:

```text
sums shape for order P = (D repeated P times..., n_bins, auxiliary...)
counts shape           = (n_bins, auxiliary...)
```

Per-pair accumulation:

```text
sums[i1, ..., iP, bin, aux...] += du[i1] * ... * du[iP]
counts[bin, aux...] += 1
```

Important documentation:

- The tensor is accumulated in the original coordinate basis.
- It does not require choosing transverse basis vectors.
- Order 2 gives `S_ij = <du_i du_j>`.
- Order 3 gives `S_ijk = <du_i du_j du_k>`.
- Higher orders are mathematically the same outer-product pattern, but only orders 2 and 3 are required for the first implementation pass.
- If bins collapse many separation directions into scalar distance bins, pairwise longitudinal/transverse projections cannot be exactly reconstructed from the binned tensor unless directional binning is also retained.
- Directional/vector separation binning is future work.

Acceptance criteria:

- Single-pair order-2 tensor equals `du * du'`.
- Single-pair order-3 tensor equals the third outer product of `du`.
- Order-2 tensor trace equals full vector second-order sum for matching bins.
- Serial and threaded tensor results match.
- Shared-position and varying-position auxiliary axes match explicit loops.

## Phase 5: Basic KHM Exact-Law Diagnostics

Add `src/KHM.jl` and expose it as `StructureFunctions.KHM`.

This phase should implement useful homogeneous/isotropic exact-law diagnostics built on scalar binned outputs and order-2/order-3 tensor outputs from Phase 4. It is not a full AGKE/KHMH budget solver.

Why not full AGKE/KHMH yet:

- The order-2 tensor `S_ij = <du_i du_j>` and order-3 tensor `S_ijk = <du_i du_j du_k>` are necessary, but not sufficient.
- Full AGKE/KHMH budgets require derivatives in separation and physical midpoint space, viscosity, time dependence or stationarity assumptions, pressure/redistribution terms, production/mean-gradient terms, and forcing/dissipation modeling or measurements.
- Implementing those terms without an explicit API for those inputs would be fake completeness.

Implement:

```julia
KHM.bin_midpoints(edges)
KHM.finite_difference(r, y)
KHM.transverse_incompressibility_residual(r, L2, T2_component; D=3)
KHM.epsilon_from_four_fifths(r, L3)
KHM.four_fifths_residual(r, L3, epsilon)
```

Definitions:

```text
T2_component_expected = L2 + r / (D - 1) * dL2/dr
transverse_incompressibility_residual = T2_component - T2_component_expected

epsilon_from_four_fifths = -5 * L3 / (4 * r)
four_fifths_residual = L3 + (4/5) * epsilon * r
```

Scope limits for this pass:

- No viscosity terms in the first pass.
- No time derivative terms in the first pass.
- No forcing terms in the first pass.
- No inhomogeneous AGKE budget terms in the first pass.
- Add placeholder types/docstrings for future full budget APIs, but do not export fake functionality.

Acceptance criteria:

- Analytic test curve for `L2(r)` produces near-zero incompressibility residual when `T2_component` is constructed from the exact relation.
- `L3(r) = -4/5 * epsilon * r` gives zero four-fifths residual to roundoff.
- KHM functions operate on plain vectors and bin edge wrappers.

## Phase 6: Operator/Result Hierarchy And Helmholtz Derived Quantities

The current code conflates two concepts:

1. Pairwise operators that can be called as `op(du, rhat)` inside the pair loop.
2. Derived structure-function quantities that require already-binned results.

Rotational and divergent second-order components from the current 2D Helmholtz
decomposition are real structure-function quantities, but they are not pairwise
operators. They are derived from binned `L2SF(r)` and `T2SF(r)` through a radial
integral relation.

Required type hierarchy:

```julia
abstract type AbstractStructureFunctionType end
abstract type AbstractPairwiseStructureFunctionType <: AbstractStructureFunctionType end
abstract type AbstractDerivedStructureFunctionType <: AbstractStructureFunctionType end
```

Rules:

- `calculate_structure_function` and `calculate_structure_function!` accept only
  `AbstractPairwiseStructureFunctionType`.
- Pairwise operators must implement `op(du, rhat)`.
- Derived operators must not implement fake pairwise call methods.
- Derived operators are evaluated by explicit derived-result APIs.

Move existing scalar pairwise operators under `AbstractPairwiseStructureFunctionType`:

```julia
SecondOrderStructureFunctionType
ThirdOrderStructureFunctionType
ProjectedStructureFunctionType
FullVectorStructureFunctionType
TransverseComponentSecondOrderStructureFunctionType
LongitudinalTransverseComponentThirdOrderStructureFunctionType
```

Add explicit derived quantity types:

```julia
RotationalSecondOrderStructureFunctionType <: AbstractDerivedStructureFunctionType
DivergentSecondOrderStructureFunctionType  <: AbstractDerivedStructureFunctionType
HelmholtzDecomposition2DType               <: AbstractDerivedStructureFunctionType
```

These types describe derived quantities. They are not valid pair-loop operators.

Public API:

```julia
helmholtz_decompose_2d(distance_bins, sums, counts)
helmholtz_decompose_2d(distance_bins, L2_sums, L2_counts, T2_sums, T2_counts)
append_helmholtz_rotational_divergent_rows(sums6, counts6, distance_bins)
```

Result type:

```julia
struct HelmholtzDecomposition2D <: AbstractStructureFunction
    distance_bins
    rotational_sums
    rotational_counts
    divergent_sums
    divergent_counts
    longitudinal_values
    transverse_values
end
```

Naming cleanup:

- Replace `postprocess_single_pass_results` with
  `append_helmholtz_rotational_divergent_rows`.
- Replace `postprocess_single_pass_2d_results` with
  `marginalize_sp2d_then_append_helmholtz_rows` or an equally explicit name.
- Do not use `postprocess` as a public API name; it does not state what is being
  computed.

Resolver behavior:

- `get_structure_function_type(2, :rotational)` returns
  `RotationalSecondOrderStructureFunction`.
- `get_structure_function_type(2, :divergent)` returns
  `DivergentSecondOrderStructureFunction`.
- Passing either derived type to `calculate_structure_function` throws a clear
  `MethodError` or `ArgumentError` explaining that Helmholtz-derived quantities
  require `helmholtz_decompose_2d`.
- A separate derived API may dispatch on derived types:

```julia
calculate_derived_structure_function(::RotationalSecondOrderStructureFunctionType, ...)
calculate_derived_structure_function(::DivergentSecondOrderStructureFunctionType, ...)
```

Acceptance criteria:

- [x] No uncallable derived quantity is accepted by pairwise calculation dispatch.
- [x] Rotational/divergent quantities are discoverable through documented public names.
- [x] The old vague `postprocess_*` names are gone from public exports.
- [x] Single-pass point-field APIs that return eight rows document rows 7 and 8 as
  Helmholtz-derived rotational/divergent rows.
- [x] Tests prove that `calculate_structure_function(RotationalSecondOrderStructureFunction, ...)`
  errors clearly, while `helmholtz_decompose_2d` succeeds from binned `L2/T2`.

## Phase 7: Documentation

Update documentation before final verification.

Files:

- `implementation_plan.md`: keep checklist current.
- `docs/theory.md`: rewrite transverse section.
- `docs/khm.md`: add basic KHM diagnostic documentation.
- `docs/helmholtz.md`: document 2D rotational/divergent decomposition.
- `README.md`: update operator table.
- `RELEASE_NOTES_v0.3.0.md` or future release notes: document breaking or semantic changes.

Required wording:

- `T2SF` means total perpendicular energy.
- `T2ComponentSF` means isotropic per-transverse-component estimate.
- `L2T1SF` and `T3SF` are signed-convention diagnostics outside 2D.
- Tensor API is the correct path for anisotropic second-order component analysis.
- Rotational/divergent second-order components are Helmholtz-derived quantities,
  not pairwise operators.
- Basic KHM diagnostics are not a full AGKE/KHMH solver.

Acceptance criteria:

- No docs describe 3D transverse direction as unique.
- No docs define transverse structure functions only by a cross product.
- Every convention-dependent operator is labeled convention-dependent.
- No public docs use `postprocess` as the name for the Helmholtz decomposition.

## Phase 8: Tests

Add or update tests.

Helper tests:

- [x] Projection identity in 2D, 3D, and helper-level D=4.
- [x] Basis orthonormality and reference-axis degeneracy behavior.
- [x] Signed 2D transverse convention matches existing orientation or documented new orientation.

Operator tests:

- [x] `S2SF = L2SF + T2SF`.
- [x] `T2ComponentSF = T2SF / (D - 1)`.
- [x] `L1T2ComponentSF = L1T2SF / (D - 1)`.
- [x] Signed odd operators match documented basis convention.

Tensor tests:

- [x] Single-pair order-2 tensor exactness.
- [x] Single-pair order-3 tensor exactness.
- [x] Order-2 tensor trace parity with full-vector second-order sums.
- [x] Auxiliary-axis tensor parity against explicit per-slice loops.
- [ ] Serial vs threaded tensor parity.

GPU tests:

- [x] KA.CPU parity for 3D invariant single-pass operators.
- [ ] KA.CPU parity for signed convention-dependent operators where supported.
- [ ] CUDA smoke/parity when CUDA is available.

KHM tests:

- [x] Analytic incompressibility residual.
- [x] Four-fifths residual.
- [x] Vector and bin-wrapper input compatibility.

Quality tests:

- [x] Aqua has no ambiguities/stale deps.
- [x] JET checks do not regress on public scalar paths.
- [ ] Allocation tests catch accidental tuple-to-matrix or padding conversions.

Helmholtz tests:

- [x] `helmholtz_decompose_2d` matches the old row-7/row-8 numerical output.
- [x] `append_helmholtz_rotational_divergent_rows` preserves rows 1 through 6 exactly.
- [x] Derived rotational/divergent types are resolvable but not accepted by pairwise calculation dispatch.
- [x] Error messages point users to `helmholtz_decompose_2d`.

## Phase 9: Verification Commands

Run these before marking the implementation complete:

```bash
julia --project=test test/test_helpers.jl
julia --project=test test/test_single_pass.jl
julia --project=test test/test_single_pass_2d.jl
julia --project=test test/test_gpu_workspace.jl
julia --project=test test/test_aqua.jl
julia --project=test test/test_jet.jl
julia --project=test test/runtests.jl
```

If CUDA is available, also run CUDA-specific smoke/parity tests.

Completion criteria:

- [x] Full local non-CUDA suite passes.
- [ ] CUDA smoke passes or is explicitly skipped because CUDA is unavailable.
- [x] `implementation_plan.md` checklist is updated with completed phases.
- [ ] Any known limitations are documented in release notes.

## Open Design Notes

These are intentionally not implementation blockers for the first pass.

- Directional/vector separation bins are needed before binned tensors can fully support anisotropic projection analysis after accumulation.
- Full AGKE/KHMH budgets require additional inputs: viscosity, time derivative information, pressure/forcing modeling or measurements, midpoint dependence, and spatial gradients.
- Higher-order tensor structure functions should be designed separately after second-order tensor storage and API are stable.
- Arbitrary `D` should not be blocked for invariant helpers, but public calculation support should only be promised where tests exist.

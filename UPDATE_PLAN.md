# StructureFunctions.jl Modernization Master Plan

Last updated: 2026-03-15
Owner: Jordan + Copilot
Status: Active working document

## Read me first

This file is the authoritative record for:

1. Current package problems.
2. Intended architecture.
3. Step-by-step implementation plan.
4. Atomic commit sequence.
5. Open decisions and acceptance criteria.

Rule: if a decision is made during implementation, write it here immediately so knowledge is not lost.

## Objectives (from user requirements)

1. Move to modern Julia package conventions for the 1.12 era:
   - Project.toml updates (compat/deps/weakdeps/extensions/workspace-aware choices).
   - Remove stale manifests where appropriate.
2. Reorganize parallel code (especially extension architecture).
3. Move NUFFT to extension(s), add clear helper APIs for our domain, verify against regular FFT, and add a slow direct exponential-sum fallback for arbitrary spacing.
4. Remove symbol-dispatch in core compute path; use types/functors with type-stable kernels.
5. Test all functions with broad coverage.
6. Add synthetic geophysical test data:
   - lat-lon box, jagged land/ocean exclusion mask, nonuniform interpolation over ocean.
   - synthetic u/v with known frequencies and validation of recovery.
7. Audit 2D-slice + time-parallel infrastructure and optimize if needed.
8. **Explicit + Qualified Imports**:
   - Move from implicit `using Package` to explicit qualified calls.
   - Rule: Use `using Package: Package [as P]` followed by `P.func`.
   - Avoid `using Package: func` for internal/hot-path code to ensure clarity.
9. **Type Relaxation**: Audit and relax overly restrictive type bounds. Prefer removing bounds entirely (e.g., remove `FT <: Real`) to allow specialization and support for Automatic Differentiation (AD) and generic number types.
10. **FixedSizeArrays Support**: Investigate `FixedSizeArrays.jl` for point-cloud representation to avoid `StaticArrays` compilation bloat for large $N$.
11. **LoopVectorization Optionality**: `LoopVectorization` is deprecated; plan to move it to an optional extension or make it toggleable to future-proof the package.

## Current package snapshot

- Top-level and test-level manifests are present.
- Project.toml currently targets older compat values.
- src includes production + prototype + deprecated variants mixed together.
- Parallel behavior exists in both src and ext.
- NUFFT logic is in src and partially incomplete.
- Tests are minimal.

## Verified findings and code issues

### A) Functional blockers and correctness

1. Hard blocker in core implementation:
   - src/Calculations.jl:103
   - `error("BadImplementationError, fix double calculating point pairs....")`
   - Impact: main tuple-based path cannot run.

2. Multiple overlapping implementations increase regression risk:
   - src/Calculations.jl
   - src/AlternateCalculations.jl
   - src/AlternateCalculations3D.jl
   - src/AlternateCalculations3D_Union_Types.jl

3. Possible third-order method wiring inconsistency across historical variants:
   - src/StructureFunctionTypes.jl needs full per-type method verification.

### B) Dispatch and type stability

1. Symbol+eval dispatch in current API helper:
   - src/StructureFunctionTypes.jl:47
   - `get_structure_function_type(x::Symbol) = eval(x)()`

2. `method::Function` fields in SF structs:
   - causes dynamic dispatch risk in hot loops
   - reduces inlining opportunities
   - weakens inference and can raise compile/runtime costs

3. Heavy union signatures in core kernels:
   - inference complexity
   - compile latency
   - harder maintenance and debugging

### C) Parallel/extension architecture

1. Deprecated parallel file still in src:
   - src/ParallelCalculations.jl

2. Extension is currently extending methods by touching another module namespace directly:
   - ext/ParallelCalculationsExt.jl
   - this needs cleaner boundaries and clearer ownership of functions

3. Worker-loading concerns are documented in code comments/logs:
   - extension appears not reliably visible on workers in current approach

### D) NUFFT implementation status

1. src/FINUFFT.jl has at least one path using `iflag`, `eps` without them in method signature.
2. src/FINUFFT_3D.jl has unfinished entry point with hard error.
3. src/FINUFFT_3D.jl has scope bug in KE branch (`eltype(u)` where `u` is undefined in scope).
4. NUFFT code currently looks prototype-level, not production-level.

### E) Test and packaging infra

1. Test coverage currently does not protect most functionality.
2. Project/test dependency and compat policy need modernization and explicit policy.
3. Manifest policy is not currently documented.

### F) Repository hygiene

1. Prototype files are mixed into src.
2. src/.ipynb_checkpoints should not be in tracked source tree.

## Target architecture

## 1) Module structure

Core package (`src`) should contain only stable code:

- Public API and exported types.
- Core serial/threaded SF kernels.
- Core type definitions and utilities.

Optional capabilities go to extensions:

- Parallel distributed extension.
- NUFFT extension.

Experimental/prototype code:

- move to experimental/ or archive/.

## 2) Dispatch design (critical)

Decision: use functor-style concrete types for SF operations.

Target pattern:

- `abstract type AbstractStructureFunctionType end`
- one concrete type per SF definition
- callable method per type: `(sf::SomeSF)(du, rhat)`
- no `method::Function` fields in hot path structs

Why:

- better inference and inlining
- avoids dynamic dispatch through abstract `Function`
- clearer semantics and easier testing

Compatibility boundary:

- string/symbol parsing may remain only in non-hot API boundary
- parser maps to concrete types without `eval` in kernels

Required validation for this refactor:

1. `@inferred` on representative SF calls.
2. `@code_warntype` clean for core kernels.
3. allocation checks before/after.

## 3) Data layout and kernel policy

Canonical internal representation (proposal):

- Normalize input into a compact internal layout at boundary functions.
- Keep hot kernels simple and strongly typed.
- Avoid giant signature unions in inner loops.

Pair traversal policy:

- use unique pairs only (`j > i`) unless explicitly requesting directed pairs
- define and test this behavior in docs and tests
- **Point representation**: Investigate if `FixedSizeArray` (from `FixedSizeArrays.jl`) provides better trade-offs than `SVector` for medium-to-large point sets where `StaticArrays` inference might struggle.

NaN policy:

- define one global rule for handling NaNs/missing points
- test all backends with this same rule

## 4) Parallel policy

- Keep one serial reference implementation as numerical truth.
- Threaded and distributed implementations must match reference within tolerance.
- Distributed extension should expose explicit setup/requirements for workers.
- Avoid silent behavior differences between extension-loaded and non-extension modes.

## 5) NUFFT policy

Implement three backend modes under one unified interface:

1. Uniform FFT reference backend.
2. FINUFFT backend.
3. Direct exponential-sum backend (slow fallback/reference for arbitrary spacing).

For every backend, specify:

- coordinate scaling conventions
- phase/sign conventions
- normalization conventions
- expected output shape/order

Numerical parity tests are mandatory.

## Phased implementation plan

## Phase 0: baseline and guardrails

Status: not started

Tasks:

1. Inventory exported API and current call paths.
2. Add baseline correctness fixtures and micro-bench script.
3. Record manifest policy decision: Remove `Manifest.toml` from the repository and rely on `Project.toml` [compat] bounds.

Exit criteria:

- baseline script and fixture references committed.

## Phase 1: unblock correctness

Status: not started

Tasks:

1. Remove hard blocker in src/Calculations.jl and implement unique-pair traversal.
2. Add pair-count regression tests (`n*(n-1)/2`).
3. Add brute-force tiny-case reference checks.
4. Audit and fix SF method wiring consistency (especially third-order signed magnitudes).
5. Add granular unit tests for `HelperFunctions.jl` (digitize, nhat, rhat, projections).

Exit criteria:

- core runs without intentional blockers
- correctness tests passing

## Phase 2: package modernization

Status: not started

Tasks:

1. Update Project.toml compat/deps/weakdeps/extensions.
2. Finalize manifest policy and remove stale manifests if policy says so.
3. Ensure test environment metadata is correct.
4. Add CI matrix including Julia 1.12.

Exit criteria:

- package resolves and tests cleanly on supported Julia versions

## Phase 3: dispatch/type-stability refactor

Status: not started

Tasks:

1. Replace `Function`-field SF structs with functors.
2. Remove eval-based symbol dispatch from hot path.
3. Introduce boundary parser to map symbols/strings to typed functors.
4. Simplify kernel signatures and input normalization.
5. Add inference/allocation tests.

Exit criteria:

- type-stable core path demonstrated by tests and warntype checks

## Phase 4: parallel cleanup and 2D-slice/time strategy

Status: not started

Tasks:

1. Define canonical parallel API.
2. Refactor extension boundary and method ownership.
3. Remove/archive deprecated parallel source path.
4. Benchmark:
   - parallel-over-time strategy
   - parallel-over-space-pairs strategy
   - hybrid chunking strategies
5. Choose and document strategy with memory/runtime tradeoffs.

Exit criteria:

- one validated parallel architecture with deterministic tests

## Phase 5: NUFFT extension and fallback

Status: not started

Tasks:

1. Move NUFFT code into extension module(s).
2. Fix known bugs and unfinished code paths.
3. Create helper API for geophysical setup (lat-lon region, masking, interpolation, scaling).
4. Implement direct exponential-sum fallback backend.
5. Add backend parity tests against FFT and direct sum references.

Exit criteria:

- NUFFT stack is optional, documented, and numerically validated

## Phase 6: synthetic dataset and full test expansion

Status: not started

Tasks:

1. Build deterministic synthetic data generator:
   - rectangular lat-lon domain
   - jagged land mask
   - ocean-only nonuniform x-y interpolation
   - u/v fields with known frequency content
2. Validate frequency recovery (FFT/NUFFT/direct sum).
3. Validate SF workflows across dimensions and SF types.
4. Add stress and edge-case tests:
   - NaNs
   - sparse masks
   - tiny/large bins
   - degenerate geometry

Exit criteria:

- broad, deterministic tests covering package functionality

## Phase 7: cleanup, docs, release prep

Status: not started

Tasks:

1. Move archive candidates out of src.
2. Update README/API docs/migration notes.
3. Add release checklist and semver change notes.

Exit criteria:

- clean tree and release-ready documentation. This includes documenting all functions with docstrings, and having examples for how to call the exported methods.

## Atomic commit plan (do one commit at a time)

Each commit below is intentionally small and reviewable.

## Block A: stabilize core

1. Commit A1: add failing regression test for blocked core path.
2. Commit A2: implement unique-pair traversal in src/Calculations.jl.
3. Commit A3: add pair-count correctness tests.
4. Commit A4: add brute-force numerical reference tests on tiny synthetic cases.
5. Commit A5: fix SF method wiring inconsistencies + logic errors in `HelperFunctions.jl` + tests.
6. Commit A6: add granular unit tests for `HelperFunctions.jl`.
7. Commit A7: convert to explicit qualified imports in core and ext.
8. Commit A8: Relax type restrictions across all core modules and extension to support AD and generic number types.

## Block B: package and infra modernization

1. Commit B1: update Project.toml compat/deps with no behavior change.
2. Commit B2: define and implement manifest policy (remove/update manifests).
3. Commit B3: refresh test environment metadata.
4. Commit B4: add CI workflow matrix for supported Julia versions.
5. Commit B5: manage `LoopVectorization` deprecation and investigate optionality/extension.

## Block C: dispatch/functor refactor

1. Commit C1: introduce functor SF types alongside existing API (temporary dual path).
2. Commit C2: switch core kernels to new functor call path.
3. Commit C3: remove `method::Function` fields.
4. Commit C4: replace eval parser with explicit boundary mapping.
5. Commit C5: add inference/allocation tests and remove legacy dispatch path.
6. Commit C6: investigate `FixedSizeArrays.jl` for optimal point-cloud representation.

## Block D: parallel architecture

1. Commit D1: add serial-vs-threaded-vs-distributed equivalence tests (initial, maybe skipped if extension missing).
2. Commit D2: refactor extension method ownership and loading behavior.
3. Commit D3: remove/archive src/ParallelCalculations.jl.
4. Commit D4: implement selected parallel chunking strategy.
5. Commit D5: add benchmark script + docs for chosen strategy.

## Block E: NUFFT extension and fallback

1. Commit E1: move NUFFT code from src to extension scaffolding.
2. Commit E2: fix currently known NUFFT bugs and remove dead entrypoints.
3. Commit E3: add unified backend API skeleton (FFT/NUFFT/direct-sum).
4. Commit E4: implement direct exponential-sum backend.
5. Commit E5: add parity tests and normalization/phase docs.

## Block F: Synthetic Dataset & Full Test Expansion

1. Commit F1: Implement Non-Uniform Domain (Mercator-ish) & Deterministic Field Generator.
2. Commit F2: Implement Tiered Spectral Verification Engine with `CairoMakie` extension for visual diagnostics.
3. Commit F3: Implement Exhaustive Input Matrix Test Factory (verifying `SVector`, `Matrix` views, `Tuple` of `Vector`s, etc.).
4. Commit F4: Implement Structure Function E2E Suite (Multi-component vector SF parity, Mixed precision (Float32/Float64), Threading stability).
5. Commit F5: Implement Deep Stability Checks (Forward-mode AD `MockDual` verification, Degenerate and Collinear point robustness).
6. Commit F6: Finalize documentation, update `README.md` and `test/plots/README.md`, and project cleanup.

## Block G: Performance and Precompilation Maximization

1. Commit G1: Cleanup any unnecessary allocations. Be as thorough as possible (replace temporary vectors with tuples, svetors, generators etc as one small example -- do anything we can to remove unnecessary allocations)
2. Commit G2: Cleanup any inference problems and define a small precompilation workload
3. Commit G3: Ensure we are not missing any type annotations. e.g. i see a lot of boolean flags that are unannotated.
4. Commit G4: Check if any of our option flags (e.g. verbose, show_progress, return_sums_and_counts, etc.) should be moved to option objects for maintainability, inheritance, and performance.

## Block H
1. Commit H1: Test if any of this can be GPU accelerated... it's pairs of points and embarassingly parallel in some ways so perhaps GPU can help. If we find that it is amenable to GPU acceleration, create a GPU backend extension and test it. This is for the Structure Functions, but potentiallly the FFTs have some GPU acceleration potential as well? Check for each backend (for FFTW and FINUFFT their documentation may have relevant information)
2. Commit H2: Benchmark performance of CPU vs GPU and provide strong and weak scaling results and figures.

## Block I: final cleanup and release prep

1. Commit I1: move archive candidates out of src.
2. Commit I2: remove tracked notebook checkpoint artifacts + ignore rules.
3. Commit I3: rewrite README with modern API examples, figures, full package description, how-tos, etc.
4. Commit I4: add migration notes and changelog draft.
5. Commit I5: version bump (0.<next minor release>) + release checklist update. (remind me to put put call juliaregistrator on the 0.2 commit on github (not sure if it can be done from command line, i dont think so, just give me the release notes) and ensure we have the julia tagbot set up.) ensure all ci is using the right julia version

## Block J: 
1. Commit J1: If everythign goes well, bump to v1.0, and have me register to the registry (provide me with the release notes)

## Atomic commit quality gate (apply before each commit)

For every commit:

1. Tests added/updated for touched behavior.
2. No unrelated refactors in same commit.
3. Public API changes documented in commit message body.
4. Inference/allocation check included for hot-path changes.
5. **Explicit Import Check**: Ensure no new implicit imports are introduced.
6. **Progress Log**: Update this plan file progress log.

## What to archive/move from src (candidate list)

- src/ParallelCalculations.jl
- src/AlternateCalculations.jl
- src/AlternateCalculations3D.jl
- src/AlternateCalculations3D_Union_Types.jl
- src/StructureFunctionTypes_old.jl
- src/FINUFFT_test.jl
- src/.ipynb_checkpoints/

## Test matrix checklist (must eventually pass)

1. Dimensions:
   - 1D, 2D, 3D supported paths
2. SF variants:
   - all exported second and third order forms
3. Inputs:
   - tuples, vectors, svectors (or normalized equivalents)
4. Binning:
   - linear and logarithmic
   - explicit bins and auto bins
5. Parallel modes:
   - serial/threaded/distributed equivalence
6. Spectral modes:
   - FFT, NUFFT, direct exponential sum parity
7. Data quality:
   - NaNs and masked regions
8. Stability/perf:
   - inference checks
   - basic allocation regression checks

## Performance and diagnostics checklist

At minimum track these for each major refactor:

1. Time per call on small/medium synthetic data.
2. Allocations per call.
3. Compile latency for first call and second call.
4. Thread scaling efficiency (1, 2, 4, 8 threads if available).
5. Distributed scaling sanity check (if workers available).

## Open decisions (must resolve early)

1. Supported Julia version window:
   - 1.12+ (Required for `[workspace]` and modern dev workflows)
2. Manifest policy for this repository:
   - No `Manifest.toml` committed in repository (enforced via `.gitignore`).
3. Canonical internal data layout choice
4. Exact tolerance policy for backend parity tests
5. Archive location strategy:
   - experimental/ vs archive/ vs separate repo
6. Public API compatibility policy:
   - preserve old names via deprecation vs direct breaking change

## Risks and mitigations

1. Numerical convention drift across refactors.
   - Mitigation: lock reference fixtures and parity tests before deep rewrites.
2. Parallel divergence from serial truth.
   - Mitigation: deterministic equivalence tests in CI.
3. NUFFT convention confusion (phase/scale/sign).
   - Mitigation: one documented normalization contract and parity checks.
4. Compile latency increase from over-general signatures.
   - Mitigation: normalize at boundaries, simplify inner kernels.
5. Scope creep due to many prototypes.
   - Mitigation: commit-by-commit gates and phase exit criteria.

## Definition of done for modernization

All items below must be true:

1. Core SF algorithms run without blockers and pass correctness tests.
2. Dispatch is type/functor-based in hot path.
3. Parallel architecture is unified and validated.
4. NUFFT is extension-based and verified against FFT/direct-sum references.
5. Synthetic geophysical test workflows exist and pass.
6. CI runs full supported matrix.
7. src contains only active production code.
8. README/docs reflect current API and workflows.

## Progress log

- 2026-03-15:
  - Initial audit completed.
  - First modernization plan drafted.
  - Plan expanded with dispatch/functor architecture details and atomic commit sequence.
  - Added rules for explicit imports and type relaxation.
  - **983520c (A1)**: Implemented a regression test in `test/test_core_correctness.jl` to verify the `BadImplementationError` (blocked path) as a baseline.
  - **870237e (A2)**: Refactored `src/Calculations.jl` to use `j > i` unique-pair traversal, removing the double-counting error and unblocking the core tuple-based path.
  - **fc27b2e (A3)**: Added verification tests in `test/test_core_correctness.jl` to ensure pair counts match the expected $N(N-1)/2$ for various point sets.
  - **f957305 (A4)**: Added numerical reference tests for a "tiny case" (2-3 points) with hand-calculated structure function values to verify mathematical accuracy.
  - **8042241 (A5)**: Fixed a wiring error in `src/StructureFunctionTypes.jl` (incorrect method for OffDiagonalConsistentThirdOrder) and a projection logic bug in `src/HelperFunctions.jl` (incorrect `n̂` call in `mδu_t`); removed the tracked `Manifest.toml` in favor of dynamic instantiation.
  - **a9439d9 (A6)**: Created `test/test_helpers.jl` for granular unit testing of `HelperFunctions.jl`. Fixed a broadcast argument swap bug in `digitize` discovered during test development.
  - **ffd751a0 (A7)**: Transitioned all modules (`HelperFunctions.jl`, `StructureFunctionTypes.jl`, `Calculations.jl`, and `ParallelCalculationsExt.jl`) to strict qualified explicit imports. Fixed precompilation blockers caused by `UndefVarError`s and circularity in extension loading. Archived prototype and deprecated files to `archive/` directory. Verified with 11/11 passing tests.
  - **dc16deb (A8)**: Relax type restrictions across all core modules and extension. Fixed `InexactError` by ensuring accumulation vectors use floating-point types. Verified with 11/11 passing tests.
  - **752052c (B1)**: Updated `Project.toml` with modern compat (Julia 1.12+) and `[workspace]` support for enhanced multi-package development.
  - **Commit B2 & B3**: Established "no manifest" policy for the library to avoid environment clutter. Refined test metadata using the new workspace structure.
  - **bb5cb62 (B4)**: Added GitHub Actions CI workflow targeting Julia 1.12 across Linux, macOS, and Windows.
  - **0dde8cf (B5)**: Documented planned `LoopVectorization` optionality/deprecation in `Calculations.jl` and `StructureFunctionTypes.jl`.
  - **6bd9e54 (C1)**: Introduced functor-style dispatch layer in `StructureFunctionTypes.jl`.
  - **0b15170 (C2)**: Switched core kernels in `Calculations.jl` to use functor dispatch.
  - **f1af625 (C3 & C4)**: Removed `.method` fields from structs and replaced `eval` with a static map-based lookup for `get_structure_function_type`.
  - **Commit 862f12c (C5)**: Added `@inferred` and allocation tests verification, and made `HelperFunctions.jl` aliases constant to fix inference. Verified with 16/16 passing tests.
  - **Commit C6 (Investigation)**: Completed `FixedSizeArrays.jl` investigation. Results showed significant performance benefits for high-dimensional point clouds (N > 50) compared to `StaticArrays.jl`.
  - **Commit e639184 (C7)**: Successfully refactored all core signatures in `HelperFunctions.jl`, `Calculations.jl`, and `ParallelCalculationsExt.jl` for generic `AbstractArray` support, eliminating restrictive `Union` type lists.
  - **Commit 6f6c82a (C8)**: Implemented a formal type-stability audit using `JET.jl`. Optimized hot loops using `ntuple` and `Val(N)` static constructors. Achieved zero-allocation paths for NTuple/SVector inputs.

## Phase 4: Parallel Cleanup (Block D) [DONE]
- **Goal**: Standardize parallel dispatch and verify mathematical equivalence across backends.
- **Milestones**:
    - [x] Commit 9b6ce5c: Integrate Aqua.jl for package hygiene (100% Compliance).
    - [x] Commit 9b6ce5c: Implement serial-vs-threaded-vs-distributed equivalence tests (Verified).
    - [x] Commit 9b6ce5c: Refactor extension method ownership (Standardized).
    - [x] Commit 9b6ce5c: Remove/archive src/ParallelCalculations.jl (Completed).

- 2026-03-16:
  - **Commit 9b6ce5c (Block D)**:
    - **Aqua.jl Integration**: Added automated hygiene checks via `Aqua.jl`, achieving 100% compliance across method ambiguities, unbound type parameters, and stale dependencies. Sanitized `Project.toml` by migrating `SharedArrays` to an extension and removing stale direct dependencies.
    - **Signature Standardisation**: Refined all core and extension signatures to use `Tuple{T, Vararg{T}}` patterns. This eliminated "smelly" type defaults and enforced explicit argument passing, while restoring support for nested indexable types (tuples-of-tuples, vectors-of-vectors) that were previously breaking in special cases.
    - **Parallel Parity Verification**: Implemented `test/test_parallel_equivalence.jl`, which verifies that serial, threaded, and distributed backends produce bit-identical numerical results and point-pair counts.
    - **Legacy Removal & Benchmarking**: Successfully removed legacy `src/ParallelCalculations.jl` and Jupyter checkpoints. Head-to-head benchmarking confirmed the new `$O(N^2/2)$` `@distributed` strategy provides a ~5.8x speedup over the legacy `$O(N^2)` `pmap` strategy, significantly reducing communication overhead.

## Phase 5: NUFFT extension and fallback (Block E) [DONE]
- **Goal**: Refactor spectral analysis into an extension-based unified API.
- **Milestones**:
    - [x] Commit 9796854: move NUFFT code from src to extension scaffolding.
    - [x] Commit 9796854: fix currently known NUFFT bugs and remove dead entrypoints.
    - [x] Commit 9796854: add unified backend API skeleton (FFT/NUFFT/direct-sum).
    - [x] Commit 9796854: implement direct exponential-sum backend.
    - [x] Commit 9796854: add parity tests and fix scaling/normalization.

## Phase 6: Synthetic Dataset & Full Test Expansion (Block F) [DONE]
- **Goal**: Implement deterministic verification engine and robust SF E2E suite.
- **Milestones**:
    - [x] Commit 9d6bd2c: add deterministic spectral signal generator and tiered verification loop.
    - [x] Commit 9d6bd2c: integrate CairoMakie for visual spectral recovery diagnostics.
    - [x] Commit 2f1a3b4: implement exhaustive input matrix test factory.
    - [x] Commit 210dc16: implement Structure Function E2E suite with threading stability fixes.
    - [x] Commit 210dc16: add deep stability checks for AD and collinear points.
    - [x] Commit 210dc16: standardize Fourier convention and finalize documentation.

- 2026-03-16:
  - **Commit 9796854 (Block E)**:
    - **Submodule Architecture**: Introduced the `SpectralAnalysis` submodule as a dedicated domain for Fourier-based analysis. This keeps the top-level `StructureFunctions` namespace clean and provides a clear separation of concerns between structure functions and spectral power density calculations.
    - **Extension-Based Backends**: Implemented `FINUFFT` and `FFTW` backends as formal Julia Extensions. This ensures that heavy spectral dependencies are only loaded when explicitly requested by the user, keeping the core package footprint minimal.
    - **Automated Scaling & Physical Recovery**: Standardized all backends to use a unified physical scaling convention. The system now automatically maps arbitrary coordinate ranges to the internal $[0, 2\pi]$ range required by NUFFT and recovers physical wavenumbers $k_{phys} = k \cdot 2\pi/L$ automatically for the user.
    - **Phase Correction**: Implemented automated phase-shift logic to account for coordinate translations (e.g., $x \to x - min(x)$). This ensures that Fourier coefficients are recovered accurately regardless of the absolute coordinate values.
    - **Numerical Parity Verified**: Successfully demonstrated numerical parity ($rtol < 10^{-7}$) between `DirectSum`, `FFT` (on uniform grids), and `FINUFFT` backends across 1D and 2D scattered datasets.
  - **Commit 9d6bd2c (Block F1-F2)**:
    - **Deterministic Verification Engine**: Implemented `test/test_phase6_spectral.jl` which validates spectral recovery against tiered difficulty levels: uniform bit-parity ($1e-12$), noisy scattered parity ($1e-10$), and sampling validation against ground truth.
    - **Visual Diagnostic Pipeline**: Integrated `CairoMakie` as a test-only extension to generate multi-panel comparison subplots. These plots automatically mark target peaks and demonstrate recovery across backends, providing instant visual feedback for mathematical correctness.
    - **Balanced Diagnostics**: Optimized diagnostic subplots by decoupling sampling density (high to avoid aliasing) from spectral resolution (moderate to ensure large, visible pixels), solving the "hidden peak" visibility issue.
    - **Scaling Standardization**: Fixed a $1/N$ scaling mismatch in the `DirectSumBackend`, ensuring bit-level agreement between explicit summation and Fourier-based backends.
  - **Commit 2f1a3b4 (Block F3)**:
    - **Fourier Sign Standardization**: Standardized the default transform convention to $e^{-ikx}$ (forward) across all backends. This ensures that physical signals $e^{imkx}$ recovered with the forward transform consistently peak at wavenumber $+k$, simplifying verification and conforming to standard Fourier library expectations.
    - **Exhaustive Input Coverage**: Implemented `test/test_phase6_inputs.jl` which validates the API against `SA.SVector` coordinates, `SubArray` views, multiple fields (NU > 1), and complex-valued fields.
    - **Robustness Fixes**: Resolved `DimensionMismatch` in `SVector` conversion and achieved bit-parity for views in `FINUFFT` and `FFTW` extensions.
  - **Commit 210dc16 (Block F4-F6)**:
    - **Structure Function E2E Suite**: Implemented `test/test_phase6_sf_e2e.jl` which validates 11+ SF types on both uniform and scattered datasets.
    - **Segfault & Threading Resolution**: Fixed a critical memory safety issue in the parallel calculation loop caused by out-of-bounds indexing in `digitize`. Stabilized multithreading by introducing `ReentrantLock` for the shared output accumulation.
    - **Deep Stability Verification**: Implemented `test/test_phase6_stability.jl` to verify AD compatibility using `MockDual` and robustness to degenerate/collinear point configurations.
    - **Visual Diagnostics**: Enhanced the `CairoMakie` extension to include input field visualizations in spectral diagnostic subplots, providing a "ground truth to recovery" verification chain.
    - **Project Cleanup**: Finalized documentation in `README.md` and `test/plots/README.md`, and updated the project modernization plan.

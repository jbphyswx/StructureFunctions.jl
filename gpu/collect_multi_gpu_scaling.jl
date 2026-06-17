#= 
    collect_multi_gpu_scaling.jl — STUB (not wired into doc assets)

True HPC **strong** and **weak** scaling on GPU requires varying the number of
GPUs (or equivalent parallel resources) while holding the problem definition fixed
(strong) or scaling total work with GPU count (weak).

This package currently runs **one full GPU** per call. The doc-asset collector
(`collect_benchmark_assets.jl`) instead measures **problem-size scaling**:
fixed hardware (1 GPU + threaded CPU), sweep N.

## When multi-GPU exists, implement here

Suggested env (example only — not implemented):

    GPU_COUNTS=1,2,4,8
    N_STRONG=20000          # strong: fixed N, vary GPU count
    N_WEAK_BASE=5000        # weak: N ∝ f(GPU count)

Expected outputs (future):

    gpu/benchmark_results/multi_gpu_strong_scaling.json
    gpu/benchmark_results/multi_gpu_weak_scaling.json
    docs/src/assets/gpu_strong_scaling.png   # fixed N, GPUs 1→8
    docs/src/assets/gpu_weak_scaling.png     # N grows with GPU count

## Run (today)

    error("multi-GPU scaling not implemented")

=#

error("""
multi-GPU strong/weak scaling is not implemented.
Use collect_benchmark_assets.jl for problem-size scaling (single GPU vs CPU).
See gpu/collect_multi_gpu_scaling.jl header for the future plan.
""")

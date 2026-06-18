# ncu + Julia (clima)

Working one-liner (user-verified on this node):

```bash
ncu --set basic --target-processes all \
    julia --project=gpu -e '
        using CUDA
        x = CUDA.ones(Float32, 1024)
        x .*= 2
        CUDA.synchronize()
    '
```

Joint 2D workload — same `ncu` flags, plus kernel filter/skip:

```bash
ncu --set basic --target-processes all \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*sf2d.*' \
    --launch-skip 3 --launch-count 1 \
    julia --project=gpu gpu/profile_joint2d_ncu.jl
```

Or `bash gpu/run_ncu_joint2d.sh` (same thing; add `-o NAME` via `NCU_OUT`).

Kernel names: `bash gpu/list_joint2d_kernel_names.sh`

#!/usr/bin/env bash
# List CUDA kernel names for joint2d ncu workload (via nsys — works headless).
#
#   bash gpu/list_joint2d_kernel_names.sh
#
# Environment: N_DIST, N_VAL, COMPILE_CELLS, VALUE_ROUTE — forwarded to profile_joint2d_ncu.jl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys not found — run: module load cuda" >&2
    exit 1
fi

OUT="${LIST_OUT:-joint2d_kern_list}"

echo "Capturing kernel names -> ${OUT}.nsys-rep"
nsys profile --trace=cuda --force-overwrite=true -o "$OUT" \
    julia --project=gpu gpu/profile_joint2d_ncu.jl

echo ""
echo "Kernel summary (use a substring for NCU_KERNEL_REGEX when NCU_PROFILE_FROM_START=on):"
nsys stats --force-export=true --report cuda_gpu_kern_sum "${OUT}.nsys-rep"

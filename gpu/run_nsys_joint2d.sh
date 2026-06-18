#!/usr/bin/env bash
# Profile joint 2D with Nsight Systems (nsys).
#
# Usage (on a GPU node):
#   cd StructureFunctions.jl
#   bash gpu/run_nsys_joint2d.sh
#
# Options via environment:
#   NSYS_OUT=joint2d_inflinear   output base name (default: joint2d)
#   VALUE_ROUTE=both             passed to profile_joint2d.jl
#   N, N_DIST, N_VAL, WARMUP, REPEAT — see gpu/profile_joint2d.jl
#
# View report (login node with GUI, or copy .nsys-rep locally):
#   nsys-ui joint2d.nsys-rep
#   nsys stats joint2d.nsys-rep    # only after a successful Julia run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${NSYS_OUT:-joint2d}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys not found — run: module load cuda" >&2
    exit 1
fi

echo "Profiling -> ${OUT}.nsys-rep"
echo "Workload: julia --project=gpu gpu/profile_joint2d.jl"

nsys profile --trace=cuda,nvtx --force-overwrite=true -o "$OUT" \
    julia --project=gpu gpu/profile_joint2d.jl

REPORT="${REPO_ROOT}/${OUT}.nsys-rep"
echo "Generated: ${REPORT}"
echo ""
echo "Headless stats (no GUI — use on login/GPU node without X11):"
echo "  nsys stats --force-export=true --report cuda_gpu_kern_sum ${OUT}.nsys-rep"
echo "  nsys stats --force-export=true --report cuda_api_sum ${OUT}.nsys-rep"
echo ""
echo "GUI (local machine with display only): nsys-ui ${OUT}.nsys-rep"

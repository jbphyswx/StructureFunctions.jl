#!/usr/bin/env bash
# Profile joint 2D tiled kernel with ncu (headless).
#
# Validated on clima: bare julia works, e.g.
#   ncu --set basic --target-processes all julia --project=gpu -e 'using CUDA; ...'
#
# Usage:
#   bash gpu/run_ncu_joint2d.sh
#   NCU_TO_STDOUT=1 bash gpu/run_ncu_joint2d.sh   # metrics to terminal (no .ncu-rep)
#   COMPILE_CELLS=exact N_DIST=50 N_VAL=50 NCU_OUT=joint2d_exact bash gpu/run_ncu_joint2d.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

command -v ncu >/dev/null 2>&1 || { echo "error: ncu not found" >&2; exit 1; }

OUT="${NCU_OUT:-joint2d_ncu}"
SET="${NCU_SET:-basic}"
SKIP="${LAUNCH_SKIP:-${PREWARM:-3}}"
COUNT="${LAUNCH_COUNT:-1}"
KERNEL_REGEX="${NCU_KERNEL_REGEX:-regex:.*sf2d.*}"
export PREWARM="${PREWARM:-3}"

NCU_LOG="$(mktemp)"
trap 'rm -f "$NCU_LOG"' EXIT

NCU_ARGS=(
    --set "$SET"
    --target-processes all
    --kernel-name-base demangled
    --kernel-name "$KERNEL_REGEX"
    --launch-skip "$SKIP"
    --launch-count "$COUNT"
)

if [[ "${NCU_VERBOSE:-}" == "1" ]]; then
    NCU_ARGS+=(--verbose)
fi

if [[ "${NCU_TO_STDOUT:-}" == "1" ]]; then
    echo "ncu → stdout (${KERNEL_REGEX} skip=${SKIP} count=${COUNT})"
    ncu "${NCU_ARGS[@]}" julia --project=gpu gpu/profile_joint2d_ncu.jl 2>&1 | tee "$NCU_LOG"
else
    NCU_ARGS+=(--force-overwrite true -o "$OUT")
    echo "ncu -o ${OUT} (${KERNEL_REGEX} skip=${SKIP} count=${COUNT})"
    ncu "${NCU_ARGS[@]}" julia --project=gpu gpu/profile_joint2d_ncu.jl 2>&1 | tee "$NCU_LOG"
fi

if grep -q 'No kernels were profiled' "$NCU_LOG"; then
    echo "error: no kernels matched filter (try NCU_TO_STDOUT=1, or list names via bash gpu/list_joint2d_kernel_names.sh)" >&2
    exit 1
fi

if [[ "${NCU_TO_STDOUT:-}" != "1" ]]; then
    echo "Generated: ${REPO_ROOT}/${OUT}.ncu-rep"
    echo "Summary:   ncu --import ${OUT}.ncu-rep --page details"
fi

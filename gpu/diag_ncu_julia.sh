#!/usr/bin/env bash
# Quick ncu smoke: CUDA broadcast (known good) then joint2d script.
#
#   bash gpu/diag_ncu_julia.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
LOG="${REPO_ROOT}/test/debug/ncu_julia_diag.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "=== $(date -Is) SLURM_JOB_ID=${SLURM_JOB_ID:-<none>} ==="
    echo "=== A: ncu CUDA broadcast (control) ==="
    ncu --set basic --target-processes all --launch-count 1 \
        julia --project=gpu -e 'using CUDA; x=CUDA.ones(Float32,1024); x.*=2; CUDA.synchronize()' 2>&1 \
        | rg 'PROF==|No kernels|ERROR' || true
    echo ""
    echo "=== B: joint2d workload to stdout ==="
    NCU_TO_STDOUT=1 NCU_VERBOSE=1 bash gpu/run_ncu_joint2d.sh 2>&1 | rg 'ncu workload|PROF==|No kernels|sf2d' || true
} > "$LOG" 2>&1

echo "Wrote ${LOG}"
tail -20 "$LOG"

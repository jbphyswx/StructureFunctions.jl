#!/usr/bin/env bash
#SBATCH --job-name=sf_uni_vs_u32
#SBATCH --output=gpu/benchmark_results/unified_vs_u32_%j.out
#SBATCH --error=gpu/benchmark_results/unified_vs_u32_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p gpu/benchmark_results
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
"${JULIA:-julia}" --project=gpu -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
"${JULIA:-julia}" -t 8 --project=gpu gpu/bench_unified_vs_u32_1d.jl

#!/usr/bin/env bash
#SBATCH --job-name=sf_slices
#SBATCH --output=gpu/benchmark_results/slices_e2e_%j.out
#SBATCH --error=gpu/benchmark_results/slices_e2e_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p gpu/benchmark_results
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
"${JULIA:-julia}" --project=gpu gpu/test_slices_e2e.jl

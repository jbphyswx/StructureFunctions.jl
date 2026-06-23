#!/usr/bin/env bash
#SBATCH --job-name=sf_e2e2d
#SBATCH --output=gpu/benchmark_results/e2e_2d_%j.out
#SBATCH --error=gpu/benchmark_results/e2e_2d_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p gpu/benchmark_results
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
"${JULIA:-julia}" --project=gpu gpu/test_e2e_2d_cuda.jl

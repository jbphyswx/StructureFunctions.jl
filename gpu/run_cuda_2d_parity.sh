#!/usr/bin/env bash
#SBATCH --job-name=sf_cuda2d
#SBATCH --output=gpu/benchmark_results/cuda2d_parity_%j.out
#SBATCH --error=gpu/benchmark_results/cuda2d_parity_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p gpu/benchmark_results
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"; nvidia-smi --query-gpu=name --format=csv,noheader || true
"${JULIA:-julia}" --project=gpu gpu/test_cuda_2d_parity.jl

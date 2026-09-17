#!/usr/bin/env bash
#SBATCH --job-name=sf_cuda_gridded
#SBATCH --output=gpu/benchmark_results/cuda_gridded_parity_%j.out
#SBATCH --error=gpu/benchmark_results/cuda_gridded_parity_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p gpu/benchmark_results
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
"${JULIA:-julia}" --project=gpu -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
"${JULIA:-julia}" -t 8 --project=gpu gpu/test_cuda_gridded_parity.jl

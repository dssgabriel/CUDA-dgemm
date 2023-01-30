#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -o results/dgemm_%J.out
#SBATCH --exclusive
#SBATCH -p short

./target/gpunode-cuda-dgemm $1

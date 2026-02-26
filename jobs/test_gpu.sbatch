#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_gpu-%j.out

module load python-3.8.12-gcc-9.4.0-qgrxbxo
module load cuda-11.6.0-gcc-9.4.0-system

source venv/bin/activate

python test_torch_gpu.py

#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/jupyter-%j.out

module load python-3.8.12-gcc-9.4.0-qgrxbxo
module load cuda-11.6.0-gcc-9.4.0-system

source ~/projects/ASD-DiagNet/venv/bin/activate

jupyter notebook --no-browser --ip=0.0.0.0 --port=8888

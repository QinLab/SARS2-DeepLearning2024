#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=distributed_learning
#SBATCH --gres=gpu:4
 
source ~/myenv/bin/activate
export PYTHONPATH="$HOME:$PYTHONPATH"
python3 main.py -var Alpha
#!/bin/bash
 
# execute in the general partition
#SBATCH --partition=gpu
 
# Number of  processes/tasks
#SBATCH --ntasks=1
 
# Run all process on a single/multiple node
#SBATCH --nodes=1
 
# Number of CPU cores per task
#SBATCH --cpus-per-task=8
 
# time
#SBATCH --time=40:00:00
 
# job name is my_job
#SBATCH --job-name=DT
 
#only use if gpu access required, 2GPUs requested
#SBATCH --gres=gpu:2
 
source activate myenv

export PYTHONPATH="$HOME:$PYTHONPATH"
# application execution
python3 ./fine_tunning.py
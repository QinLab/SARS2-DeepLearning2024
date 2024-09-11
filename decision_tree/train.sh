#!/bin/bash
 
# execute in the general partition
#SBATCH --partition=firefly-full
 
# Number of  processes/tasks
#SBATCH --ntasks=1
 
# Run all process on a single/multiple node
#SBATCH --nodes=1
 
# Number of CPU cores per task
#SBATCH --cpus-per-task=32
 
# time
#SBATCH --time=40:00:00
 
# job name is my_job
#SBATCH --job-name=DT
 
#only use if gpu access required, 2GPUs requested
#SBATCH --gres=gpu:4

module load anaconda
source activate myenv

export PYTHONPATH="$HOME:$PYTHONPATH"
# application execution
python3 ./train.py
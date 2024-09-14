#!/bin/bash
 
# execute in the general partition
#SBATCH --partition=epyc-full
 
# Number of  processes/tasks
#SBATCH --ntasks=1
 
# Run all process on a single/multiple node
#SBATCH --nodes=1
 
# Number of CPU cores per task
#SBATCH --cpus-per-task=16
 
# time
#SBATCH --time=3-00:00:00
 
# job name is my_job
#SBATCH --job-name=Alpha-test
 
#only use if gpu access required, 2GPUs requested
#SBATCH --gres=gpu:2
 
# load environment
module load anaconda
source activate myenv

export PYTHONPATH="$HOME:$PYTHONPATH"
# application execution
python3 ./local_shap.py -Rand -var Alpha

#!/bin/bash
 
# execute in the general partition
#SBATCH --partition=firefly-full
 
# Number of  processes/tasks
#SBATCH --ntasks=1
 
# Run all process on a single/multiple node
#SBATCH --nodes=1
 
# Number of CPU cores per task
#SBATCH --cpus-per-task=80
 
# time
#SBATCH --time=3-00:00:00
 
# job name is my_job
#SBATCH --job-name=Delta
 
#only use if gpu access required, 2GPUs requested
#SBATCH --gres=gpu:4
 
# load environment
module load anaconda
source activate myenv

export PYTHONPATH="$HOME:$PYTHONPATH"
# application execution
python3 ./global_shap.py -var Delta

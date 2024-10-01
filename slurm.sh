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
#SBATCH --time=3-00:00:00
 
# job name is my_job
#SBATCH --job-name=venn_shap_num1500

# Set the name of the output file
#SBATCH --output=./outs/venn_shap_num1500_%j.out
 
#only use if gpu access required, 2GPUs requested
#SBATCH --gres=gpu:2
 
# load environment
module load anaconda
source activate myenv

export PYTHONPATH="$HOME:$PYTHONPATH"
# python3 ./training/data_prep.py
# python3 ./training/split_data_train_val_test.py
# python3 ./training/main.py -pkl -batch_size 64 -epochs 15
# python3 ./first_sequences/group_fasta.py
# python3 ./first_sequences/save_first_sequences.py
# python3 ./first_sequences/label_first_sequences.py
# python3 ./gene_shap/single_shap.py -init -var Omicron
# python3 ./gene_shap/single_shap.py -rand -var Omicron
# python3 ./gene_shap/single_shap.py -ID_shapvalue EPI_ISL_1641648
# python3 ./gene_shap/overall_shap.py -var Gamma
# python3 ./gene_shap/correlation.py
python3 ./gene_shap/venn_shap.py -num 1500 -agg
# python3 ./gwas/chi_square_test.py
# python3 ./gwas/gwas_vs_shap.py
# python3 ./gwas/venn_gwas_who.py -num 1500 -agg -allvar
# python3 ./decision_tree/data_dt.py
# python3 ./decision_tree/fine_tunning.py
# python3 ./decision_tree/shap_dt.py -model cat
# python3 ./decision_tree/train.py
# python3 ./gene_shap/msa.py -var Alpha -num 1000 -thr 900
# python3 ./gene_shap/venn_dna.py -freq 20000


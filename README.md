# Explainable Convolutional Neural Network Model Provides an Alternative Genome-Wide Association Perspective on Mutations in SARS-CoV-2  

Code repository of the paper https://arxiv.org/abs/2410.22452 

## Overview
![Summary of or model](/results/model.png)
We compared an explainable CNN and traditional GWAS to identify SARS-CoV-2 mutations linked to WHO-labeled Variants of Concern (VOCs). The CNN, combined with SHAP, outperformed GWAS in identifying key nucleotide substitutions, particularly in spike gene regions. Our findings highlight the potential of explainable neural networks as an alternative to traditional genomic analysis methods.


## Setup
1. Create a Virtual Environment
```
python3 -m venv myenv
```
2. Activate the Virtual Environment
```
source myenv/bin/activate
```
3. Install Packages from `requirements.txt`
```
pip3 install -r requirements.txt
```
4. If you want to run on cluster use `slurm.sh`:
    1. Update the SLURM directives (e.g., #SBATCH options) to match the specifications of your cluster. 
    2. Activate your virtual environment: ```source activate myenv```
    3. Uncomment the scripts you want to execute: For example, to run the Venn SHAP calculation, you would leave this line uncommented: ```python3 ./gene_shap/venn_shap.py -num 1500 -agg```
    4. Submit the job: ```sbatch slurm.sh```
    **Notes:** Output Files: The script will create an output file in the `./outs/` directory
5. If you want to run without cluster:
    1. Activate the environment with: ```source myenv/bin/activate```
    2. Run the desired Python script from root of repository, for example: ```python3 ./gene_shap/venn_shap.py -num 1500 -agg```

## Repository Structure
1. Train model: `./training/`
2. Find initial sequences in each VOCs: `./first_sequences/`
3. Analysis SHAP value for two-layer CNN model: `./gene_shap/`
4. Compare SHAP value with GWAS: `./gwas/`
5. Train decision tree models and produce their SHAP values: `./decision_tree/`

** Optional: Find MSA within each variant: `./msa_gene/`

**Note:** All results will be saved in `./results/`

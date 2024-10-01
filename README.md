# Explainable Convolutional Neural Network Model Provides an Alternative Genome-Wide Association Perspective on Mutations in SARS-CoV-2  

Code repository of the paper <link_of_paper>

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



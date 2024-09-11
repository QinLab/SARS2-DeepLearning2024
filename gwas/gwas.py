import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as CONST
import json
import numpy as np
import pandas as pd
from utils_gwas import *
from utils_gene import *


#-----------------Analyze p-value-----------------
df_pvalue = pd.read_csv('./p_value_csv/p_values_corrected.csv')
df_pvalue = norm_log_p_values(df_pvalue)

df_orfs = pd.read_csv(CONST.ORf_DIR)
df_snv_ORF = map_snp_to_orf(df_pvalue, df_orfs)

# 1- The distribution of p-value across different ORFs
plot_dist_value_across_gene(df_snv_ORF, 'sum_of_norm_logp_by_orf')

# 2- What percentage of the highest pvalue is associated with each gene
count_greatest_pvalue = (df_snv_ORF['norm(-logp)'] == 1).sum()
print(f"Number of positions with value 1 in 'norm(-logp)': {count_greatest_pvalue}")

P_value_pos = df_snv_ORF[df_snv_ORF['norm(-logp)'] == 1]
# P_value_pos['Position'] = P_value_pos.index.astype(int)
orf_percentage = P_value_pos['ORF'].value_counts(normalize=True)*100
'''
ORF2 (S)    29.34%
ORF1ab      24.6%
ORF1a       23.02%
ORF5 (M)    16.28%
Non_ORF      2.79%
ORF9 (N)     1.79%
ORF8         0.86%
ORF3a        0.57%
ORF6         0.28%
ORF4 (E)     0.14%
ORF7a        0.14%
ORF7b        0.14%
'''

#-----------------Analyze SHAP value-----------------
agg_shap_allVariants = mean_agg_shap_values()
df_agg_ORF = map_snp_to_orf(agg_shap_allVariants, df_orfs)

df_agg_ORF_merge = pd.merge(df_agg_ORF, agg_shap_allVariants, on='Position', how="inner")

# 1- The distribution of SHAP across different ORFs
plot_dist_value_across_gene(df_agg_ORF_merge, 'sum_of_norm_shap_by_orf')

# 2- What percentage of the highest SHAP values is associated with each gene
count_greatest_shapvalue = (df_agg_ORF_merge['norm-Aggregation'] == 1).sum()
great_num = max(count_greatest_shapvalue, count_greatest_pvalue)

greatest_SHAP_value_pos = df_agg_ORF_merge.nlargest(great_num, 'norm-Aggregation')
orf_percentage = greatest_SHAP_value_pos['ORF'].value_counts(normalize=True) * 100
'''
ORF2 (S)    55.16%
ORF1ab      20.08%
ORF1a       11.54%
ORF8         6.16%
ORF9 (N)     3.87%
Non_ORF      2.36%
ORF7a        0.71%
ORF5 (M)     0.07%
'''
#-----------------GWAS vs SHAP-----------------
with open('mutations.json', 'r') as json_file:
    mutations_data = json.load(json_file)

mutation_alpha = mutations_data["mutation_alpha"]
mutation_beta = mutations_data["mutation_beta"]
mutation_gamma = mutations_data["mutation_gamma"]
mutation_delta = mutations_data["mutation_delta"]
mutation_omicron = mutations_data["mutation_omicron"]

df_mutation_alpha = pd.DataFrame(mutation_alpha, columns=['Nucleotides and Positions', 'Condon', 'Genes'])
position_spike_alpha = extract_positions(mutation_alpha)
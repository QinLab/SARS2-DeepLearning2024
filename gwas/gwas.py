import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as CONST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils_gwas import *


df_pvalue = pd.read_csv('./p_value_csv/p_values_corrected.csv')
df_pvalue = norm_log_p_values(df_pvalue)

df_orfs = pd.read_csv(CONST.ORf_DIR)
df_snv_ORF = map_snp_to_orf(df_pvalue, df_orfs)

#-----------------Analyze p-value-----------------
# 1- The distribution of p-value across different ORFs
plot_dist_pvalue_across_gene(df_snv_ORF)

# 2- What percentage of the highest pvalue is associated with each gene
count_heighest_pvalue = (df_snv_ORF['norm(-logp)'] == 1).sum()
print(f"Number of positions with value 1 in 'norm(-logp)': {count_heighest_pvalue}")

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


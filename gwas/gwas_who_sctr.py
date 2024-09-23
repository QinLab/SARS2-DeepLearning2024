import sys
import os
import constants.constants as CONST
import numpy as np
import pandas as pd
from utils_gwas import *



if __name__ == "__main__":
    #-----------------Analyze p-value-----------------
    df_pvalue = pd.read_csv('./p_value_csv/p_values_chi_square.csv')
    df_pvalue['Positions'] = df_pvalue['Positions'].astype(int)
    df_pvalue = norm_log_p_values(df_pvalue)

    df_orfs = pd.read_csv(CONST.ORf_DIR)
    df_snv_ORF = map_snp_to_orf(df_pvalue, df_orfs)

    # 1- The distribution of p-value across different ORFs
    plot_dist_value_across_gene(df_snv_ORF, 'sum_of_norm_logp_by_orf', '-log(p-value)')

    # 2- What percentage of the highest pvalue is associated with each gene
    count_greatest_pvalue = (df_snv_ORF['Normalized -log(p-value)'] == 1).sum()
    print(f"Number of positions with value 1 in 'Normalized -log(p-value)': {count_greatest_pvalue}")

    greatest_pvalue_pos = df_snv_ORF[df_snv_ORF['Normalized -log(p-value)'] == 1]
    # greatest_pvalue_pos['Position'] = greatest_pvalue_pos.index.astype(int)
    orf_percentage = greatest_pvalue_pos['ORF'].value_counts(normalize=True)*100
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
    agg_shap_allVariants['Positions'] = agg_shap_allVariants['Positions'].astype(int)
    df_agg_ORF = map_snp_to_orf(agg_shap_allVariants, df_orfs)

#     df_agg_ORF_merge = pd.merge(df_agg_ORF, agg_shap_allVariants, on='Positions', how="inner")

    # 1- The distribution of SHAP across different ORFs
    plot_dist_value_across_gene(df_agg_ORF, 'sum_of_norm_shap_by_orf', 'SHAP value')

    # 2- What percentage of the highest SHAP values is associated with each gene
    count_greatest_shapvalue = (df_agg_ORF['Normalized SHAP value'] == 1).sum()
    print(f"Number of positions with value 1 in 'Normalized SHAP value': {count_greatest_shapvalue}")
    great_num = max(count_greatest_shapvalue, count_greatest_pvalue)
    print("great_num:", great_num)

    greatest_SHAP_value_pos = df_agg_ORF.nlargest(great_num, 'Normalized SHAP value')
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
    stack_all_variant = match_gwas_shap(agg_shap_allVariants, df_pvalue, df_orfs)

    # Plot all shap regarding genes
    df_agg_ORF_merge_aa = pd.merge(df_agg_ORF, stack_all_variant, on='Positions', how='outer')
    plot_by_genes(df_agg_ORF_merge_aa, 'Normalized SHAP value')

    # Plot all -log(p-value) regarding genes
    df_snv_ORF_plot = pd.merge(df_snv_ORF, stack_all_variant, on='Positions', how='outer')
    plot_by_genes(df_snv_ORF_plot, "Normalized -log(p-value)")
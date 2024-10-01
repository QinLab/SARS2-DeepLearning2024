import argparse
import constants as CONST
import numpy as np
import os
import pandas as pd
from gwas.utils_gwas import *

arg_parser = argparse.ArgumentParser(description="GWAS vs SHAP")
arg_parser.add_argument("-perc", "--percentage", type=int, default = None,
                        help="Percentage of rows to select based on the highest values (e.g., 10 for top 10%).")
arg_parser.add_argument("-num", "--number", type=int, default = None,
                        help="Number of rows to select based on the highest values")
args = arg_parser.parse_args()
perc = args.percentage
num = args.number


if __name__ == "__main__":
    #-----------------Analyze p-value-----------------
    df_pvalue = pd.read_csv(CONST.PVLU_DIR2)
    df_pvalue['Positions'] = df_pvalue['Positions'].astype(int)
    df_pvalue = norm_log_p_values(df_pvalue)

    df_orfs = pd.read_csv(CONST.ORf_DIR)
    df_snv_ORF = map_snp_to_orf(df_pvalue, df_orfs)

    # 1- The distribution of p-value across different ORFs
    plot_dist_value_across_gene(df_snv_ORF, 'sum_of_norm_logp_by_orf_test', '-log(p-value)')
    
    # 2- What percentage of the highest pvalue is associated with each gene
    if perc:
        greatest_pvalue_pos = df_snv_ORF.nlargest(int((perc/100)*len(df_snv_ORF)), 'Normalized -log(p-value)')
        info_pvalue = f"Top {perc}% value in pvalue"
        
    elif num:
        greatest_pvalue_pos = df_snv_ORF.nlargest(num, 'Normalized -log(p-value)')
        info_pvalue = f"Top {num} pvalue"
        
    else:   
        count_greatest_pvalue = (df_snv_ORF['Normalized -log(p-value)'] == 1).sum()
        greatest_pvalue_pos = df_snv_ORF[df_snv_ORF['Normalized -log(p-value)'] == 1]
        info_pvalue = f"{count_greatest_pvalue} normalized(-log(p-value)) has value 1"
        
    # greatest_pvalue_pos['Position'] = greatest_pvalue_pos.index.astype(int)
    orf_percentage_pvalue = greatest_pvalue_pos['ORF'].value_counts(normalize=True)*100
    print(f'{info_pvalue}: {orf_percentage_pvalue}')
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

    # 1- The distribution of SHAP across different ORFs
    plot_dist_value_across_gene(df_agg_ORF, f'sum_of_norm_shap_by_orf_test', 'SHAP value')

    # 2- What percentage of the highest SHAP values is associated with each gene
    if perc:
        greatest_SHAP_value_pos = df_agg_ORF.nlargest(int((perc/100)*len(df_agg_ORF)), 'Normalized SHAP value')
        info_shap = f"Top {perc}% value in normalized aggregated SHAP values"
        
    elif num:
        greatest_SHAP_value_pos = df_agg_ORF.nlargest(num, 'Normalized SHAP value')
        info_shap = f"Top {num} normalized aggregated SHAP values" 

    else:   
        count_greatest_shapvalue = (df_agg_ORF['Normalized SHAP value'] == 1).sum()
        greatest_pvalue_pos = df_agg_ORF[df_agg_ORF['Normalized SHAP value'] == 1]
        great_num = max(count_greatest_shapvalue, count_greatest_pvalue)
        greatest_SHAP_value_pos = df_agg_ORF.nlargest(great_num, 'Normalized SHAP value')
        info_shap = f"{count_greatest_shapvalue} normalized aggregated SHAP values has value 1"
        
    orf_percentage_shap = greatest_SHAP_value_pos['ORF'].value_counts(normalize=True) * 100
    print(f'{info_shap}: {orf_percentage_shap}')
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
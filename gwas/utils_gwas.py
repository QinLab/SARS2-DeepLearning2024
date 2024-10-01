import constants.constants as CONST
from collections import Counter
from gene_shap.utils_shap import get_var_shap_count, count_common_positions_all_combinations, get_commonset_who_shap
from gwas.utils_gene import *
import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import os
import pandas as pd
from random import uniform
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler


def split_sequence(sequence):
    return list(sequence)


def replace_invalid_chars(char):
    valid_chars = 'acgtn-'
    if char not in valid_chars:
        return 'i'
    else:
        return char


def calculate_p_value(df, column):
    # Create a contingency table
    contingency_table = pd.crosstab(index=df[column],
                                    columns=df['Variant_VOC']) 
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p


def get_great_pvalue_position_orf(df_snv_ORF, df_orfs, num=None, thr=None, perc=None):

    pvalue_position_orf = []
    df_snv_ORF =df_snv_ORF.drop_duplicates(subset='Positions', keep='first')
    count_greatest_pvalue = (df_snv_ORF['Normalized -log(p-value)'] == 1).sum()
    
    if perc:
        greatest_pvalue_pos = df_snv_ORF.nlargest(int((perc/100)*len(df_snv_ORF)), 'Normalized -log(p-value)')
        info_pvalue = f"Top {perc}% value in pvalue"
    elif thr:
        greatest_pvalue_pos = df_snv_ORF[df_snv_ORF['Normalized -log(p-value)'] >= thr]
        info_pvalue = f"pvalues with values higher than {thr}"
    elif num:
        if num < count_greatest_pvalue:
            num = count_greatest_pvalue
        greatest_pvalue_pos = df_snv_ORF.sort_values(by='Normalized -log(p-value)', ascending=False).iloc[:num, :]
        info_pvalue = f"Top {num} pvalues"        
    else:          
        greatest_pvalue_pos = df_snv_ORF[df_snv_ORF['Normalized -log(p-value)'] == 1]
        info_pvalue = f"{count_greatest_pvalue} normalized(-log(p-value)) has value 1"

    for _, row_pvalue in greatest_pvalue_pos.iterrows():
        for _, row in df_orfs.iterrows():
            key = row_pvalue['Positions']
            key = int(key)            
            matching_gene = 'Non_ORF'
            if key >= int(row['Start']) and key <= int(row['End']):
                matching_gene = row['Gene']
                break
                
        key_new = f'{key} ({ matching_gene})'
        pvalue_position_orf.append(key_new)

    var_df = pd.DataFrame(pvalue_position_orf, columns=['Positions'])
    return var_df, count_greatest_pvalue


def get_commonset_who_pvalue(agg, common_all_var=None, num=None, thr=None, perc=None):
    
    if num:
        name_plot= f"num_{num}"
        num_pos = num
    elif thr:
        name_plot= f"thr_{thr}"
    elif perc:
        name_plot= f"perc_{perc}"
        num_pos = int((perc/100) * CONST.SEQ_SIZE)
    else:
        name_plot = "default_plot"
        
    df_orfs = pd.read_csv(CONST.ORf_DIR)
    df_pvalue = pd.read_csv(CONST.PVLU_DIR2)
    set_names = CONST.VOC_WHO
    
    df_pvalue['Positions'] = df_pvalue['Positions'].astype(int)
    df_pvalue = norm_log_p_values(df_pvalue)
    df_snv_ORF = map_snp_to_orf(df_pvalue, df_orfs)
    p_value_pos, count_greatest_pvalue = get_great_pvalue_position_orf(df_snv_ORF, df_orfs, num=num, thr=thr, perc=perc)
    p_value_pos = p_value_pos['Positions']
    
    if num==None and thr==None and perc==None: # when we do not define any criteria
        num = count_greatest_pvalue
        
    all_sets, agg = get_commonset_who_shap(thr, num, perc, agg)
    
    common_set, all_values, combinations_with_counts = count_common_positions_all_combinations(all_sets, set_names)
    
    set_pvalue = set(p_value_pos)
    if common_all_var or thr:
        name_plot = f"{name_plot}_common"
        set_common_set = set(common_set['Alpha, Beta, Gamma, Delta, Omicron'][0])

    else:
        flattened_values = [val for sublist in all_values for val in sublist]
        counter = Counter(flattened_values)
        most_common = counter.most_common(num_pos)
        set_common_set = {value for value, count in most_common}
    
    return set_pvalue, set_common_set, agg, name_plot
    

def norm_log_p_values(df_pvalue):
    """
    Processes the p-values in the DataFrame to compute the -log(p) values, handle infinities, and normalize them.

    Returns:
    DataFrame: The input DataFrame with additional columns '-logp' and 'Normalized -log(P-value)'.
    """
    df_pvalue['-logp'] = -np.log(df_pvalue['P-Value'])
    
    # Find the maximum finite value in the '-logp' column where it is not infinity
    max_finite = np.nanmax(df_pvalue.loc[df_pvalue['-logp'] != np.inf, '-logp'])

    df_pvalue['-logp'].replace(np.inf, max_finite, inplace=True)

    # Normalize the '-logp' values using MinMaxScaler
    scaling_pvalue = MinMaxScaler()
    df_pvalue['Normalized -log(p-value)'] = scaling_pvalue.fit_transform(df_pvalue[['-logp']])
    
    return df_pvalue


def map_snp_to_orf(df, df_orfs):
    """
    Maps SNP positions to corresponding ORFs or marks them as 'Non_ORF' if they do not match any ORF.
    """
    df_value_ORF = pd.DataFrame(columns=['Positions', 'ORF'])
    df_orfs['Start'] = df_orfs['Start'].astype(int)
    df_orfs['End'] = df_orfs['End'].astype(int)
    
    for idx, row in df.iterrows():
        pos = row['Positions']

        # A mask to check if the SNP position falls within any ORF
        mask = (pos >= df_orfs['Start']) & (pos <= df_orfs['End'])

        if mask.any():
            # Extract the matching 'Gene' values
            matching_genes = df_orfs.loc[mask, 'Gene'].tolist()

            matching_df = pd.DataFrame({'Positions': [pos] * len(matching_genes),
                                        'ORF': matching_genes})
        else:
            matching_df = pd.DataFrame({'Positions': [pos],
                                        'ORF': "Non_ORF"})

        df_value_ORF = pd.concat([df_value_ORF, matching_df], ignore_index=True)

    df_value_ORF = df_value_ORF.merge(df, on='Positions', how='inner')

    return df_value_ORF


def plot_dist_value_across_gene(df, file_name, value):
    
    grouped_dfs = {}
    for orf, group_df in df.groupby('ORF'):
        grouped_dfs[orf] = group_df
    
    # Calculate the sum of 'Normalized -log(P-value)' for each ORF group and store it in a dictionary
    sums_by_orf = {orf: grouped_dfs[f'Normalized {value}'].sum() for orf, grouped_dfs in grouped_dfs.items()}

    sorted_sums_by_orf = dict(sorted(sums_by_orf.items(), key=lambda item: item[1], reverse=True))

    sorted_orfs = list(sorted_sums_by_orf.keys())
    sorted_sums = list(sorted_sums_by_orf.values())

    # Calculate the total sum of values for percentage calculation
    total_sum = sum(sorted_sums)
    percentage_dict = {key: (value / total_sum) * 100 for key, value in sorted_sums_by_orf.items()}

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_orfs, sorted_sums, color='skyblue')
    plt.xlabel('ORF')
    plt.ylabel(f'Sum of normalized {value}')
    plt.title(f'Distribution of normalized {value} over ORF (Descending Order)')
    plt.xticks(rotation=45)

    for bar, orf in zip(bars, sorted_orfs):
        height = bar.get_height()
        percentage = percentage_dict[orf]
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%', ha='center', va='bottom')
    
    directory_path = f"{CONST.RSLT_DIR}/gwas_shap_plot"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(f'{directory_path}/{file_name}.png', dpi=140, bbox_inches='tight')


def mean_agg_shap_values(heatmap=False):
    """
    Processes SHAP values for VOCs, computes the mean, merges them into a single DataFrame,
    and normalizes the aggregation.
    """
    merged_df = pd.DataFrame()
    variants = CONST.VOC_WHO
    for variant in variants:
        df = pd.read_csv(f'{CONST.SHAP_DIR}/agg_{variant}_beeswarm.csv')
        
        # Compute the mean, excluding the last row (aggregation of all shap values) and column (VOCs name)
        df_mean = df.iloc[:-1, :-1].mean()
        
        variant_df = pd.DataFrame(df_mean, columns=[variant])
        
        if merged_df.empty:
            merged_df = variant_df
        else:
            merged_df = merged_df.merge(variant_df, left_index=True, right_index=True)
            
    if heatmap:
        return merged_df
    else:
        # Compute the mean aggregation across all variants
        merged_df['mean-Aggregation'] = merged_df.mean(axis=1)

        scaling_SHAP = MinMaxScaler()
        merged_df['Normalized SHAP value'] = scaling_SHAP.fit_transform(merged_df[['mean-Aggregation']])

        agg_shap_allVariants = merged_df[['Normalized SHAP value']].copy()
        agg_shap_allVariants['Positions'] = agg_shap_allVariants.index

        return agg_shap_allVariants


def get_aminoacide_pvalue_shap(mutation_data, variant_name, agg_shap_allVariants, df_pvalue, df_orfs):
    """
    Processes mutation data for a specific variant and merges it with additional data.
    """
    df_mutation = pd.DataFrame(mutation_data, columns=['Nucleotides and Positions', 'Amino Acide Changes', 'Genes'])

    position_spike = extract_positions(df_mutation, df_orfs)
    df_position_spike = pd.DataFrame(position_spike, columns=['Nucleotides and Positions', 'Amino Acide Changes', 'Genes', 'Positions'])
    df_position_spike = df_position_spike.explode('Positions', ignore_index=True)

    # Merge with additional data
    merged_shap_pvalue = df_position_spike.merge(agg_shap_allVariants, on='Positions', how='left')
    merged_shap_pvalue = merged_shap_pvalue.merge(df_pvalue, on='Positions', how='left')

    result_df = merged_shap_pvalue[["Positions", "Amino Acide Changes", "Genes", "Normalized SHAP value", "Normalized -log(p-value)"]]
    result_df['Variant'] = variant_name

    return result_df


def match_gwas_shap(agg_shap_allVariants, df_pvalue, df_orf):
    
    with open(CONST.MUT_DIR, 'r') as json_file:
        mutations_data = json.load(json_file)

    variants = {
        "Alpha": mutations_data["mutation_alpha"],
        "Beta": mutations_data["mutation_beta"],
        "Gamma": mutations_data["mutation_gamma"],
        "Delta": mutations_data["mutation_delta"],
        "Omicron": mutations_data["mutation_omicron"]
    }

    variant_dfs = []
    for variant_name, mutation_data in variants.items():
        variant_df = get_aminoacide_pvalue_shap(mutation_data, variant_name, agg_shap_allVariants, df_pvalue, df_orf)
        variant_dfs.append(variant_df)

    # Concatenate all variants into a single DataFrame
    stacked_all_variant = pd.concat(variant_dfs, axis=0)
    stacked_all_variant['AminoAcid'] = stacked_all_variant['Amino Acide Changes'] + "(" + stacked_all_variant['Variant'] + ")"
    stacked_all_variant = stacked_all_variant[["Positions","Amino Acide Changes", "AminoAcid"]]
    
    return stacked_all_variant


def plot_by_genes(df, norm):
    
    WHO_spike = ["H69-", "V70-", "Y144-", "N501Y", "A570D", "D614G",
                "P681H", "T716I", "S982A", "D1118H", "D80A", "D215G", "L241-", "L242-", 
                "A243-", "K417N", "E484K", "A701V", "L18F", "T20N", "P26S", "D138Y",
                "R190S", "K417T", "H655Y", "T1027I", "V1176F", "L452R", "P681R",
                "E484Q", "Q107H", "T19R", "T478K", "D950N", "F157-", "E156-"]
    
    genes = ["ORF1ab","ORF1a", "ORF2 (S)", "ORF3a", "ORF4 (E)", 
             "ORF5 (M)", "ORF6", "ORF7a", "ORF7b", "ORF8", "ORF9 (N)", "ORF10", "Non_ORF"]
    
    color_map = {"ORF1ab":"violet", "ORF1a":"lightcoral","ORF2 (S)": "red",
                 "ORF3a": "lavender", "ORF4 (E)": "cyan", "ORF5 (M)": "pink",
                 "ORF6": "olive", "ORF7a":'brown', "ORF7b":"gray",
                 "ORF8": "green", "ORF9 (N)": "blue", "ORF10":"greenyellow",
                "Non_ORF":"orange", "WHO_spike":"black"}
    
    fig, ax = plt.subplots(figsize=(13.33,7.5), dpi = 96)
    
    # Create the grid 
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
    for gene in genes:
        df_color = df[df['ORF']==gene]
        colors = []
        for _, row in df_color.iterrows():
            colors.append(color_map[gene])
        ax.scatter(df_color['Positions'], df_color[norm], c=colors, label=gene)
    
    colors_who = []
    pos_who = []
    aa_who = []
    for aa in WHO_spike:
        for _, row_aa in df.iterrows():
            if row_aa["Amino Acide Changes"] == aa:
                colors.append("black")
                pos_who.append(row_aa['Positions'])
                aa_who.append(row_aa[norm])
            
    ax.scatter(pos_who, aa_who, c="black", label="WHO_spike")
    
    # Remove the spines
    ax.spines[['top','left','bottom']].set_visible(False)
    
    ax.set_xlabel('Positions', fontsize = 14)
    if norm == "Normalized SHAP value":
        ax.set_ylabel('Normalized SHAP value', fontsize = 14)
    else:
        ax.set_ylabel(f'{norm}', fontsize = 14)
    ax.plot([0.12, .9], [.98, .98], transform=fig.transFigure, clip_on=False, color='red', linewidth=.6)
    ax.add_patch(plt.Rectangle((0.12,.98), 0.04, -0.02, facecolor='red', transform=fig.transFigure, clip_on=False, linewidth = 0))

    if norm == "Normalized SHAP value": 
        ax.legend(loc="best", ncol=1, bbox_to_anchor=[1, 1.07], borderaxespad=0, frameon=False, fontsize=12)
    # Add in title and subtitle
    if norm == "Normalized SHAP value":
        norm = "Normalized aggregation of SHAP value"
    ax.text(x=0.12, y=.93, s=f"Scatter plot of {norm} within", transform=fig.transFigure, ha='left', fontsize=14, weight='bold', alpha=.8)
    ax.text(x=0.12, y=.90, s=" Alpha, Beta, Delta, Gamma, Omicron", transform=fig.transFigure, ha='left', fontsize=14,weight='bold', alpha=.8)
    
    directory_path = f"{CONST.RSLT_DIR}/gwas_shap_plot"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(f'{directory_path}/manhattan_{norm}_test.png', dpi=140, bbox_inches='tight')

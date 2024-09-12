import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as CONST
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
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


def map_snp_to_orf(df_pvalue, df_orfs):
    """
    Maps SNP positions to corresponding ORFs or marks them as 'Non_ORF' if they do not match any ORF.
    """
    df_snv_ORF = pd.DataFrame(columns=['Positions', 'ORF'])
    df_orfs['Start'] = df_orfs['Start'].astype(int)
    df_orfs['End'] = df_orfs['End'].astype(int)
    
    for idx, row in df_pvalue.iterrows():
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

        df_snv_ORF = pd.concat([df_snv_ORF, matching_df], ignore_index=True)

    df_snv_ORF = df_snv_ORF.merge(df_pvalue, on='Positions', how='inner')

    return df_snv_ORF


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
    
    output_path = f'{CONST.RSLT_DIR}/gwas-plot/{file_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def mean_agg_shap_values():
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

    # Compute the mean aggregation across all variants
    merged_df['mean-Aggregation'] = merged_df.mean(axis=1)

    scaling_SHAP = MinMaxScaler()
    merged_df['Normalized SHAP value'] = scaling_SHAP.fit_transform(merged_df[['mean-Aggregation']])

    agg_shap_allVariants = merged_df[['Normalized SHAP value']].copy()
    agg_shap_allVariants['Positions'] = agg_shap_allVariants.index

    return agg_shap_allVariants


def get_aminoacide_pvalue_shap(mutation_data, variant_name, agg_shap_allVariants, df_pvalue):
    """
    Processes mutation data for a specific variant and merges it with additional data.
    """
    df_mutation = pd.DataFrame(mutation_data, columns=['Nucleotides and Positions', 'Amino Acide Changes', 'Genes'])

    position_spike = extract_positions(df_mutation)
    df_position_spike = pd.DataFrame(position_spike, columns=['Nucleotides and Positions', 'Amino Acide Changes', 'Genes', 'Positions'])
    df_position_spike = df_position_spike.explode('Positions', ignore_index=True)

    # Merge with additional data
    merged_shap_pvalue = df_position_spike.merge(agg_shap_allVariants, on='Positions', how='left')
    merged_shap_pvalue = merged_shap_pvalue.merge(df_pvalue, on='Positions', how='left')

    result_df = merged_shap_pvalue[["Positions", "Amino Acide Changes", "Genes", "Normalized SHAP value", "Normalized -log(p-value)"]]
    result_df['Variant'] = variant_name

    return result_df


def match_gwas_shap(agg_shap_allVariants, df_pvalue):
    
    with open('mutations.json', 'r') as json_file:
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
        variant_df = get_aminoacide_pvalue_shap(mutation_data, variant_name, agg_shap_allVariants, df_pvalue)
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
        ax.scatter(df_color['Position'], df_color[norm], c=colors, label=gene)
    
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
    
    ax.set_xlabel('Position', fontsize = 14)
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
    ax.text(x=0.12, y=.90, s=" Alpha, Beta, Delta,Gamma,Omicron", transform=fig.transFigure, ha='left', fontsize=14,weight='bold', alpha=.8)
    
    output_path = f'./test2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


def plot_venn_digram_shap_pvalue(num, df_snv_ORF, df_agg_ORF_merge):
    
    greatest_pvalue_pos = df_snv_ORF.nlargest(num, 'Normalized -log(p-value)')
    greatest_SHAPvalue_pos = df_agg_ORF_merge.nlargest(num, 'Normalized SHAP value')
    greatest_pvalue_pos['Position'] = greatest_pvalue_posp['Position'].astype(int)

    greatest_SHAPvalue_pos['position'] = greatest_SHAPvalue_pos['position'].astype(int)

    positions_pvalue = set(greatest_pvalue_pos['Position'])
    positions_shapvalue = set(greatest_SHAPvalue_pos['position'])

    # Compute the intersection
    common_values = positions_pvalue & positions_shapvalue
    print(f"Number of common values: {len(common_values)}")

    plt.figure(figsize=(8, 6))
    venn = venn2([positions_pvalue, positions_shapvalue], 
                 set_labels=('P-value Positions', 'SHAP Value Positions'))

    venn.get_label_by_id('10').set_text(f'{len(positions_pvalue - positions_shapvalue)}')
    venn.get_label_by_id('01').set_text(f'{len(positions_shapvalue - positions_pvalue)}')
    venn.get_label_by_id('11').set_text(f'{len(common_values)}')

    # Set title and display the plot
    plt.title(f'Number of Common Positions within {num} greatest value in p-value and SHAP value')
    
    output_path = f'./venn.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
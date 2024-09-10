import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as CONST
import pandas as pd
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
    DataFrame: The input DataFrame with additional columns '-logp' and 'norm(-logp)'.
    """
    df_pvalue['-logp'] = -np.log(df_pvalue['P-Value'])
    
    # Find the maximum finite value in the '-logp' column where it is not infinity
    max_finite = np.nanmax(df_pvalue.loc[df_pvalue['-logp'] != np.inf, '-logp'])

    df_pvalue['-logp'].replace(np.inf, max_finite, inplace=True)

    # Normalize the '-logp' values using MinMaxScaler
    scaling_pvalue = MinMaxScaler()
    df_pvalue['norm(-logp)'] = scaling_pvalue.fit_transform(df_pvalue[['-logp']])
    
    return df_pvalue


def map_snp_to_orf(df_pvalue, df_orfs):
    """
    Maps SNP positions to corresponding ORFs or marks them as 'Non_ORF' if they do not match any ORF.
    """
    df_snv_ORF = pd.DataFrame(columns=['Position', 'ORF'])

    for idx, row in df_pvalue.iterrows():
        pos = row['Position']

        # A mask to check if the SNP position falls within any ORF
        mask = (pos >= df_ORFs['Start']) & (pos <= df_ORFs['End'])

        if mask.any():
            # Extract the matching 'Gene' values
            matching_genes = df_ORFs.loc[mask, 'Gene'].tolist()

            matching_df = pd.DataFrame({'Position': [pos] * len(matching_genes),
                                        'ORF': matching_genes})
        else:
            matching_df = pd.DataFrame({'Position': [pos],
                                        'ORF': "Non_ORF"})

        df_snv_ORF = pd.concat([df_snv_ORF, matching_df], ignore_index=True)

    df_snv_ORF = df_snv_ORF.merge(df_pvalue, on='Position', how='inner')

    return df_snv_ORF


def plot_dist_pvalue_across_gene(df):
    
    grouped_dfs = {}
    for orf, group_df in df_snv_ORF.groupby('ORF'):
        grouped_dfs[orf] = group_df

    # Calculate the sum of 'norm(-logp)' for each ORF group and store it in a dictionary
    sums_by_orf = {orf: grouped_dfs['norm(-logp)'].sum() for orf, grouped_dfs in grouped_dfs.items()}

    sorted_sums_by_orf = dict(sorted(sums_by_orf.items(), key=lambda item: item[1], reverse=True))

    sorted_orfs = list(sorted_sums_by_orf.keys())
    sorted_sums = list(sorted_sums_by_orf.values())

    # Calculate the total sum of values for percentage calculation
    total_sum = sum(sorted_sums)

    # Calculate the percentage for each ORF 
    percentage_dict = {key: (value / total_sum) * 100 for key, value in sorted_sums_by_orf.items()}

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_orfs, sorted_sums, color='skyblue')
    plt.xlabel('ORF')
    plt.ylabel('Sum of norm(-logp)')
    plt.title('Sum of norm(-logp) by ORF (Descending Order)')
    plt.xticks(rotation=45)

    # Add percentage text on top of each bar
    for bar, orf in zip(bars, sorted_orfs):
        height = bar.get_height()
        percentage = percentage_dict[orf]
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.2f}%', ha='center', va='bottom')
        
    output_path = f'{CONST.RSLT_DIR}/p-values-plot/sum_of_norm_logp_by_orf.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

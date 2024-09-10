import pandas as pd
from scipy.stats import chi2_contingency


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
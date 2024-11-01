import constants.constants as CONST
from gwas.utils_gwas import split_sequence, replace_invalid_chars, calculate_p_value
import numpy as np
import os
import pandas as pd
from scipy.stats import chi2_contingency


df_train = pd.read_csv(CONST.TRAIN_DIR)
df_test = pd.read_csv(CONST.TEST_DIR)
df_concatenated = pd.concat([df_train, df_test], ignore_index=True)
df = df_concatenated[["sequence", "Variant_VOC"]]
del df_concatenated


if __name__ == '__main__':
    sequences = df['sequence'].apply(lambda x: split_sequence(x))
    processed_sequences = np.array(sequences.tolist())
    df_seq = pd.DataFrame(processed_sequences, columns=np.arange(1, 29892))
    df_seq = df_seq.applymap(replace_invalid_chars)
    # get the phynotype for chi-squre test
    df_merged = pd.concat([df_seq, df["Variant_VOC"]], axis=1)

    # Calculate p-value for each position with the corrected function
    p_values = {column: calculate_p_value(df_merged, column) for column in df_merged.columns[:-1]}

    p_values_df = pd.DataFrame(p_values.items(), columns=['Positions', 'P-Value'])

    directory_path = CONST.PVLU_DIR
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    p_values_df.to_csv(f'{directory_path}/p_values_chi_square_test.csv', index=False)
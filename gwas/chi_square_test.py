import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as CONST
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from utils_gwas import split_sequence, replace_invalid_chars, calculate_p_value



df_train = pd.read_csv(CONST.TRAIN_DT)
df_test = pd.read_csv(CONST.TEST_DT)
df_concatenated = pd.concat([df_train, df_test], ignore_index=True)
df = df_concatenated[["sequence", "Variant_VOC"]]
del df_concatenated

sequences = df['sequence'].apply(lambda x: split_sequence(x))
processed_sequences = np.array(sequences.tolist())
df_seq = pd.DataFrame(processed_sequences, columns=np.arange(1, 29892))
df_seq = df_seq.applymap(replace_invalid_chars)
# get the phynotype for chi-squre test
df_merged = pd.concat([df_seq, df["Variant_VOC"]], axis=1)

# Calculate p-value for each position with the corrected function
p_values = {column: calculate_p_value(df_merged, column) for column in df_merged.columns[:-1]}

p_values_df = pd.DataFrame(p_values.items(), columns=['Positions', 'P-Value'])
p_values_df.to_csv('./p_value_csv/p_values_chi_square.csv', index=False)
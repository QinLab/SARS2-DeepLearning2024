import argparse
import numpy as np
import pandas as pd
import sars.constants as CONST
import shap
from shap_functions import *
import tensorflow as tf
from plots.waterfall import waterfall


# Argument Parsing
arg_parser = argparse.ArgumentParser(description="SHAP Value Calculation for SARS-CoV-2 Variants")
arg_parser.add_argument("-num_seq_basevalue", "--num_seq_basevalue", type=int, default=5,
                        help="Number of sequences to produce base value with (default=5)")
arg_parser.add_argument("-initial", "--initial_seq", type=bool, default=True,
                        help="Shap value for initial sequence or another sequence")
arg_parser.add_argument("-ID_basevalue", "--ID_basevalue", default=None,
                        help="List of IDs to use for base value sequences (default=None)")
arg_parser.add_argument("-ID_shapvalue", "--ID_shapvalue", default=None,
                        help="List of ID to use for SHAP value sequences (default=None)")
arg_parser.add_argument("-var", "--variant", 
                        help="The name of varient you want explore its SHAP value. 'None' if you want to produce SHAP value for a single sequence with its id")


args = arg_parser.parse_args()
num_seq = args.num_seq_basevalue
initial = args.initial_seq
ID_basevalue = args.ID_basevalue
ID_shapvalue = args.ID_shapvalue
var = args.variant
variants = CONST.VOC_WHO


# Load Data
df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
df_test = pd.read_csv(f'{CONST.TEST_DIR}')
df_train_test = pd.concat([df_train, df_test])
first_sequences = pd.read_csv(f'{CONST.FRST_DIR}/first_detected.csv')

if initial:
    df_initial = pd.read_csv(f'{CONST.FRST_DIR}/first_detected.csv')
    df_sequences = first_sequences[first_sequences['Variant_VOC'] == var]
else:
    df_sequences = df_train_test[df_train_test["ID"] == ID_shapvalue]

# # Convert reference sequences to one-hot encoded format and get its genetic sequence with lower case
ref_seq_oneHot, converted_sequence = convert_ref_to_onehot_lowercase()


#---------------------------Base value---------------------------
# Get some sequences from test dataset to produce base value
base_features = []

for variant in variants:
    features = calculate_base_value(df_train, variant, args.num_seq_basevalue, args.ID_basevalue)
    base_features.append(features)

combined_features = np.concatenate(base_features, axis=0)

# Load model
model = tf.keras.models.load_model(f'{CONST.MODEL_SAVE}')

# Calculate base value
explainer = shap.DeepExplainer(model, combined_features)
base_value = np.array(explainer.expected_value)


#---------------------------shap value for initial sequences of each variant---------------------------
index = CONST.VOC_WHO.index(var)
shap_values = calculate_shap_value(model, explainer, base_value, var, df_sequences, ID_shapvalue, index)

##---------------------------Get the positions as column names of each shap value---------------------------
df_ORFs = pd.read_csv(CONST.ORf_DIR)
df_var = get_pos_nuc(df_sequences, var, converted_sequence, df_ORFs)

# Get waterfall plot
for i in variants:
    print(f'If the {var} sequence gets classified as {i}:')
    index_var = variants.index(i)
    waterfall(var,i ,df_var, shap_values[index_var][0], base_value[index_var], 1,max_display=6, show=True)

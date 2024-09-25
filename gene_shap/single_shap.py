import argparse
import constants.constants as CONST
from decision_tree.utils_dt import get_column_names
import glob
import numpy as np
import pandas as pd
from plots.waterfall import waterfall
from plots.viz import plot_DNA
from plots.scatter import scatter_plot_var
import shap
from utils_shap import *
import tensorflow as tf


# Argument Parsing
arg_parser = argparse.ArgumentParser(description="SHAP Value Calculation for SARS-CoV-2 Variants")
arg_parser.add_argument("-num_seq_basevalue", "--num_seq_basevalue", type=int, default=5,
                        help="Number of sequences for base value (default: 5)")
arg_parser.add_argument("-init", "--initial_seq", action='store_true',
                        help="Use initial sequences for SHAP value")
arg_parser.add_argument("-ID_basevalue", "--ID_basevalue", default=None,
                        help="IDs for base value sequences (default: None)")
arg_parser.add_argument("-ID_shapvalue", "--ID_shapvalue", default=None,
                        help="ID for SHAP value sequences (default: None)")
arg_parser.add_argument("-rand", "--random", action='store_true',
                        help="SHAP value a random sequences (default: None)")
arg_parser.add_argument("-max_display", "--max_display", type=int, default=6,
                        help="Number of top SHAP values to display (default: 6)")
arg_parser.add_argument("-var", "--variant", default=None,
                        help="Variant name for SHAP value calculation; (default: None)")


args = arg_parser.parse_args()
max_display = args.max_display
num_seq = args.num_seq_basevalue
initial = args.initial_seq
ID_basevalue = args.ID_basevalue
ID_shapvalue = args.ID_shapvalue
random = args.random
var = args.variant
variants = CONST.VOC_WHO


# Load Data
df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
df_test = pd.read_csv(f'{CONST.TEST_DIR}')
df_train_test = pd.concat([df_train, df_test])
first_sequences = pd.read_csv(f'{CONST.FRST_DIR}/first_detected.csv')
df_sequences = pd.DataFrame()

# SHAP value for initial sequences of each variant
if initial:
    df_sequences = first_sequences[first_sequences['Variant_VOC'] == var]
    ID = df_sequences["ID"].values[0]
    
# SHAP value for a certain sequences with its GISAID id
if ID_shapvalue:
    df_sequences = df_train_test[df_train_test["ID"] == ID_shapvalue]
    if df_sequences.empty:
        if CONST.NM_CPU == None:
            num_cores = mp.cpu_count()  # Get the number of CPU cores
        else:
            num_cores = CONST.NM_CPU
        filenames = glob.glob(f"{CONST.BATCH_DIR}/group_*.fasta")
        args = [(filename, ID_shapvalue) for filename in filenames]
        
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(read_single_seq, args)
    
        valid_results = [res for res in results if res is not None]       
        combined_df = pd.concat(valid_results, ignore_index=True)
        df_sequences =  find_single_label(combined_df) 
    var = df_sequences['Variant_VOC'].values[0]
    ID_shapvalue = [ID_shapvalue]
                                  
# SHAP value for a random sequence from a certain variant
if random:
    df_sequences = df_train_test[df_train_test['Variant_VOC'] == var].sample(n=1)
    ID = df_sequences["ID"].values[0]


if __name__ == "__main__":
       
    # Convert reference sequences to one-hot encoded format and get its genetic sequence with lower case
    ref_seq, ref_seq_oneHot, converted_sequence = convert_ref_to_onehot_lowercase()

    #---------------------------Base value---------------------------
    # Get some sequences from test dataset to produce base value
    base_features = []

    for variant in variants:
        features = calculate_base_value(df_train, variant, num_seq, ID_basevalue)
        base_features.append(features)

    combined_features = np.concatenate(base_features, axis=0)

    # Load model
    model = tf.keras.models.load_model(f'{CONST.MODEL_SAVE}')

    # Calculate base value
    explainer = shap.DeepExplainer(model, combined_features)
    base_value = np.array(explainer.expected_value)


    #---------------------------shap value for initial sequences of each variant---------------------------
    index = CONST.VOC_WHO.index(var)
    df, features, shap_values = calculate_shap_value(model, explainer, base_value, var, df_sequences, ID_shapvalue, index)

    #---------------------------Get the positions as column names of each shap value---------------------------
    df_ORFs = pd.read_csv(CONST.ORf_DIR)
    df_var = get_pos_nuc(df_sequences, ref_seq, df_ORFs)


    #---------------------------Draw plots for SHAP value of a sequence---------------------------
    # waterfall plot
    for i in variants:
        print(f'If the {var} sequence gets classified as {i}:')
        index_var = variants.index(i)
        # with summation
        waterfall(var,i ,df_var, shap_values[index_var][0], base_value[index_var], 1,max_display=max_display, show=True)
        #without summation
        seq = df_sequences['sequence'].values[0]
        column_names = get_column_names(seq)
        df_sum = get_pos_nuc_summation(df_sequences, df_ORFs, column_names, features)
        waterfall(var, i, df_sum, shap_values[index_var],
           base_value[index_var], 1, max_display=6, summation=False, show=True)

    # Vis plot
    for i in variants:
        print(f'If the {var} sequence gets classified as {i}:')
        Id = df_sequences["ID"].iloc[0]
        DNA_obj = plot_DNA(var=var,as_var=i, Id=Id, num_shap = 5)
        DNA_obj.plot_ref_and_sequence_mutation_positions(df,
                                                 shap_values,
                                                 features,
                                                 seq_num = 0,
                                                 ref_seq = converted_sequence,
                                                 ref_seq_oneHot=ref_seq_oneHot)

    # Scatter plot of shap value along sequence
    for as_var in variants:
        scatter_plot_var(Id, shap_values, var, as_var)
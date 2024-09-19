from agg_shap import Agg_SHAP
import argparse
import numpy as np
import pandas as pd
import os
import constants.constants as CONST
import shap
import tensorflow as tf

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-var", "--variant", help="variant name")

args = arg_parser.parse_args()
var = args.variant

features_tot = []
var_who = CONST.VOC_WHO

if __name__ == '__main__':
    'Data'
    model = tf.keras.models.load_model(f'{CONST.MODEL_SAVE}')
    
    # Using df_train for producing base value
    df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
    df_test = pd.read_csv(f'{CONST.TEST_DIR}')
    
    # All of sequences, both train and test sequences, within each variant.
    df_concatenated = pd.concat([df_train, df_test], ignore_index=True)
    non_dup_test = df_concatenated.drop_duplicates(subset=['ID'], keep=False)
    
    'Calculating base value'
    # Producing base value for calculating of SHAP value for all variants
    for i in range(len(var_who)):
        var_base = var_who[i]
        calc_base = Agg_SHAP(df_train, var_base)
        _, features, _ = calc_base.get_features( num_seq = 5, 
                                             ID = None, 
                                             )
        features_tot.append(np.array(features))
            
    features_base = np.concatenate((features_tot), axis=0)
    
    explainer = shap.DeepExplainer(model, features_base)
    base_value = np.array(explainer.expected_value)
    np.savetxt(f'{CONST.SHAP_DIR}/base_{var}.csv', base_value, delimiter=',')
    
    'RAM needs to be clean'
    del df_train, df_test, model, df_concatenated, features_base
    
    'Calculating SHAP value for each variant'
    calc_shap = Agg_SHAP(non_dup_test, var)
    _, features, Ids = calc_shap.get_features( num_seq = None,
                                            ID = None, 
                                            )
    del non_dup_test
    
    shap_no_zero, non_zero_IDs = calc_shap.get_non_zero_shap_values(
        explainer, 
        np.array(features), 
        Ids,
        )
    
    'Producing a DataFrame for SHAP values. Each column name represents the position number of a nucleotide in the genetic sequence.'
    df = calc_shap.get_shap_for_each_nuc(shap_no_zero, non_zero_IDs)

    df.loc[f'Total_SHAP_{var}'] = df.sum(numeric_only=True)
    
    directory_path = f'{CONST.SHAP_DIR}/{var}_glob.csv'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    df.to_csv(directory_path, index=False)

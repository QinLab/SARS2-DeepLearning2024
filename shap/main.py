from agg_shap import Agg_SHAP
import argparse
import numpy as np
import pandas as pd
import os
import sars.constants as CONST
import shap
import tensorflow as tf

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-var", "--variant", help="variant name")

args = arg_parser.parse_args()
var = args.variant

features_tot = []
var_who = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron']

if __name__ == '__main__':
    home_dir = os.path.expanduser('~')
    model = tf.keras.models.load_model(f'{home_dir}/{CONST.MODEL_SAVE}')

    df_train = pd.read_csv(f'{home_dir}/{CONST.TRAIN_DIR}')
    df_test = pd.read_csv(f'{home_dir}/{CONST.TEST_DIR}')


    df_concatenated = pd.concat([df_train, df_test], ignore_index=True)
    non_dup_test = df_concatenated.drop_duplicates(subset=['ID'], keep=False)
    
    calc_base = Agg_SHAP(df_train, var)
    
    # Producing base value for calculating of SHAP value for all variants
    for i in range(len(var_who)):
            var_base = var_who[i]
            features,_ = calc_base.get_features( num_seq = 5, 
                                                 ID = None, 
                                                 )
            features_tot.append(np.array(features))
            
    features_base = np.concatenate((features_tot), axis=0)
    
    explainer = shap.DeepExplainer(model, features_base)
    base_value = np.array(explainer.expected_value)
    np.savetxt(f'{CONST.SHAP_DIR}/base_value_{var}.csv', base_value, delimiter=',')
    
    calc_shap = Agg_SHAP(non_dup_test, var)
    features, Ids = calc_shap.get_features( num_seq = None,
                                            ID = None, 
                                            )
    
    # RAM needs to be clean
    del df_train, df_test, model, df_concatenated, non_dup_test, 

    shap_no_zero, non_zero_IDs = cal_shap.get_non_zero_shap_values(
        explainer, 
        features, 
        Id_test,
        )

    df = aggSHAP.get_shap_for_each_nuc(shap_no_zero, non_zero_IDs)

    df.loc[f'Total_SHAP_{var}'] = df.sum(numeric_only=True)

    df.to_csv(f'{CONST.SHAP_DIR}/agg_shap{var}_beeswarm.csv', 
                    index=False)

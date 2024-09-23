import argparse
import constants.constants as CONST
import pandas as pd
import os
from decision_tree.utils_dt import *
shap.initjs()

arg_parser = argparse.ArgumentParser(description="SHAP Value Calculation for SARS-CoV-2 Variants in Decision Trees")
arg_parser.add_argument("-model", "--model",
                        help="Decision tree model for SHAP values (rf, xgb, cat)")

args = arg_parser.parse_args()
model = args.model
variants = CONST.VOC_WHO


if model == 'rf':
    model_dt = CONST.MODEL_RF
elif model=='xgb':
    model_dt = CONST.MODEL_XGB
elif model == 'cat':
    model_dt = CONST.MODEL_XGB
    
force_dir = f"{CONST.RSLT_DIR}/force_plot"
if not os.path.exists(force_dir):
    os.makedirs(force_dir)    
    
if __name__  == '__main__':
    df_initial = pd.read_csv(f'{CONST.FRST_DIR}/first_detected.csv') 
    for var in variants:
        index_var = variants.index(var)
        seq_df = df_initial[df_initial['Variant_VOC'] == var]
        explainer, shap_values, choosen_instance, df_shap = get_shap_instance(model_dt,
                                                                              seq_df,
                                                                              var)
        #with summation over the axis 2
        reshaped_array = shap_values[index_var].reshape(1, 29891, 7)
        summed_array = reshaped_array.sum(axis=2)
        filtered_df = choosen_instance.loc[:, (choosen_instance == 1).any()]
        f = shap.force_plot(explainer.expected_value[index_var], summed_array,
                        filtered_df, figsize=(30, 5), show=False)
        shap.save_html(f"{force_dir}/force_plot_summation_{var}_{model}.htm", f)

        #without summation over the axis 2
        f = shap.force_plot(explainer.expected_value[index_var], shap_values[index_var],
                        df_shap, figsize=(40, 6), show=False)
        shap.save_html(f"{force_dir}/force_plot_{var}_{model}.htm", f)
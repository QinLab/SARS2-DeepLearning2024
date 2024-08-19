import aggSHAP
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-var", "--variant", help="variant name")

args = arg_parser.parse_args()
var = args.variant

features_tot = []


if __name__ == '__main__':
    model = tf.keras.models.load_model('./weight_acc_loss_Nov_21/WHO_Nov_21_epoch_15.hdf5')

    df_train = pd.read_csv('/home/qxy699/Data/WHO_representative_random/WHO_training_BetaTest.csv')
    df_test = pd.read_csv('/home/qxy699/Data/WHO_representative_random/who_dataset_test.csv')


    df_concatenated = pd.concat([df_train, df_test], ignore_index=True)
    non_dup_test = df_concatenated.drop_duplicates(subset=['ID'], keep=False)
    
    cal_shap = Agg_SHAP(df, var)
    
    # Producing one-time base value for calculating of SHAP value for all variants
    for i in range(len(var_who)):
            var_base = var_who[i]
            features = cal_shap.get_features(df_train, 
                                            5, 
                                            ID = None, 
                                            var_base_value = var_base, 
                                                            )
            features_tot.append(features)
            
    features_base = np.concatenate((features_tot), axis=0)

    explainer = shap.DeepExplainer(model, features_base)
    base_value = np.array(explainer.expected_value)
    np.savetxt(f'./agg_shap_value/base_value_{var}.csv', base_value, delimiter=',')
    
    features, Ids = cal_shap.get_features_and_labels(explainer, 
                                                    num_seq = None
                                                    ID = None, 
                                                    )
    
    # RAM needs to be clean
    del df_train
    del df_concatenated
    del non_dup_test

    shap_no_zero, non_zero_IDs = cal_shap.get_non_zero_shap_values(
        explainer, 
        features, 
        Id_test,
        )

    df = aggSHAP.get_shap_for_each_nuc(shap_no_zero, non_zero_IDs)

    df.loc[f'Total_SHAP_{var}'] = df.sum(numeric_only=True)

    df.to_csv(f'./agg_shap_value/agg_shap{var}_beeswarm.csv', 
                    index=False)

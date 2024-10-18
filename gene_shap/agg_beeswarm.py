import constants.constants as CONST
from gene_shap.plots.beeswarm import beeswarm
from gene_shap.utils_agg_shap import Agg_SHAP as aggshap
from gene_shap.utils_shap import convert_ref_to_onehot_lowercase, orf_column_names
import pandas as pd
import numpy as np


df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
df_test = pd.read_csv(f'{CONST.TEST_DIR}')
df_train_test = pd.concat([df_train, df_test])
variants = CONST.VOC_WHO


if __name__ == "__main__":
    ref_seq, ref_seq_oneHot, converted_sequence = convert_ref_to_onehot_lowercase()
    
    for variant in variants:
        agg_var = pd.read_csv(f'{CONST.SHAP_DIR}/agg_{variant}_beeswarm.csv')
        shap_values = agg_var[:-1].iloc[:, :-1].values
        ID_list = agg_var['Accession ID'].iloc[:-1].tolist()
        
        shap_instance = aggshap(df=df_train_test, var=variant)
        _, features_test, _ = shap_instance.get_features(num_seq=None, ID=ID_list)
        
        indices = np.argmax(features_test, axis=2)
        df_bees = pd.DataFrame(indices)
        df_bees = orf_column_names(df_bees, ref_seq)
        beeswarm(variant ,agg_var.iloc[:, :-1], shap_values, df_bees)
    
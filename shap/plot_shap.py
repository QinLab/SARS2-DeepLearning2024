from agg_shap import Agg_SHAP as aggshap
import argparse
import numpy as np
import pandas as pd
from plots.viz import plot_DNA
from plots.waterfall import waterfall
from plots.scatter_orf import scatter_plot_var
import sars.constants as CONST
from sars.one_hot.one_hot import one_hot_encode_label as onehot
import shap
from shap_functions import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tqdm.autonotebook import tqdm
from tqdm import trange


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-num_seq_basevalue", "--num_seq_basevalue", type=int, default=5, help="number of sequence that we want to produce base value with (default=5)")
arg_parser.add_argument("-ID_basevalue", "--ID_basevalue", default=None, help="get some random sequence. if you want to get sequences with cetain IDs you should pass them as list to this arg (default=None. produce shap value for initial sequence within each variant)")
arg_parser.add_argument("-ID_shapvalue", "--ID_shapvalue", default=None, help="get some random sequence. if you want to get sequences with cetain IDs you should pass them as list to this arg (default=None. produce shap value for initial sequence within each variant)")

args = arg_parser.parse_args()
num_seq = args.num_seq_basevalue
ID_basevalue = args.ID_basevalue
ID_shapvalue = args.ID_shapvalue


# Get sequences
df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
df_test = pd.read_csv(f'{CONST.TEST_DIR}')
df_train_test = pd.concat([df_train, df_test])
first_sequences = pd.read_csv(f'{CONST.FRST_DIR}/first_detected.csv')

# Convert ref sequences to onehot version and get its genetic sequence with lower case
ref_seq_oneHot, converted_sequence = convert_ref_to_onehot_lowercase()


#---------------------------Base value---------------------------
# Get some sequences from test dataset to produce base value
"Alpha"
alpha_shap = aggshap(df=df_test, var="Alpha")
_,features_alpha ,_ = alpha_shap.get_features(num_seq, ID_basevalue)

"Beta"
beta_shap = aggshap(df=df_test, var="Beta")
_,features_beta ,_ = beta_shap.get_features(num_seq, ID_basevalue)

"Gamma"
gamma_shap = aggshap(df=df_test, var="Gamma")
_,features_gamma ,_ = gamma_shap.get_features(num_seq, ID_basevalue)

"Delta"
delta_shap = aggshap(df=df_test, var="Delta")
_,features_delta ,_ = delta_shap.get_features(num_seq, ID_basevalue)

"Omicron"
omicron_shap = aggshap(df=df_test, var="Omicron")
_,features_omicron,_ = omicron_shap.get_features(num_seq, ID_basevalue)


combined_features = np.concatenate((features_alpha, 
                                    features_beta, 
                                    features_delta, 
                                    features_gamma, 
                                    features_omicron), axis=0)

# Load model
model = tf.keras.models.load_model(f'{CONST.MODEL_SAVE}')

# Calculate base value
explainer = shap.DeepExplainer(model, combined_features)
base_value = np.array(explainer.expected_value)


#---------------------------shap value for initial sequences of each variant---------------------------
#------------shap value for a single sequence of alpha------------
var = "Alpha"
index = (CONST.VOC_WHO).index(var)

if ID_shapvalue != None:
    alpha = df_train_test[df_train_test['Variant_VOC']==var]
else:
    alpha = first_sequences[first_sequences['Variant_VOC']==var]
    
agg_alpha = aggshap(df=alpha, var=var)
filtered_df_alpha, features_alpha,_ = agg_alpha.get_features(1, ID_shapvalue)
shap_values_alpha = explainer.shap_values(np.array(features_alpha),
                                          check_additivity=True)

label_alpha = onehot(var)
check_additivity(model, shap_values_alpha, label_alpha, features_alpha, base_value[index])

#------------shap value for a single sequence of beta------------
var = "Beta"
index = (CONST.VOC_WHO).index(var)

if ID_shapvalue != None:
    beta = df_train_test[df_train_test['Variant_VOC']==var]
else:
    beta = first_sequences[first_sequences['Variant_VOC']==var]
    
agg_beta = aggshap(df=beta, var=var)
filtered_df_beta, features_beta,_ = agg_beta.get_features(1, ID_shapvalue)
shap_values_beta = explainer.shap_values(np.array(features_beta),
                                          check_additivity=True)

label_beta = onehot(var)
check_additivity(model, shap_values_beta, label_beta, features_beta, base_value[index])

#------------shap value for a single sequence of gamma------------
var = "Gamma"
index = (CONST.VOC_WHO).index(var)

if ID_shapvalue != None:
    gamma = df_train_test[df_train_test['Variant_VOC']==var]
else:
    gamma = first_sequences[first_sequences['Variant_VOC']==var]
    
agg_gamma = aggshap(df=gamma, var=var)
filtered_df_gamma, features_gamma,_ = agg_gamma.get_features(1, ID_shapvalue)
shap_values_gamma = explainer.shap_values(np.array(features_gamma),
                                          check_additivity=True)

label_gamma = onehot(var)
check_additivity(model, shap_values_gamma, label_gamma, features_gamma, base_value[index])

#------------shap value for a single sequence of delta------------
var = "Delta"
index = (CONST.VOC_WHO).index(var)

if ID_shapvalue != None:
    delta = df_train_test[df_train_test['Variant_VOC']==var]
else:
    delta = first_sequences[first_sequences['Variant_VOC']==var]
    
agg_delta = aggshap(df=delta, var=var)
filtered_df_delta, features_delta,_ = agg_delta.get_features(1, ID_shapvalue)
shap_values_delta = explainer.shap_values(np.array(features_delta),
                                          check_additivity=True)

label_delta = onehot(var)
check_additivity(model, shap_values_delta, label_delta, features_delta, base_value[index])

#------------shap value for a single sequence of omicron------------
var = "Omicron"
index = (CONST.VOC_WHO).index(var)

if ID_shapvalue != None:
    omicron = df_train_test[df_train_test['Variant_VOC']==var]
else:
    omicron = first_sequences[first_sequences['Variant_VOC']==var]
    
agg_omicron = aggshap(df=omicron, var=var)
filtered_df_omicron, features_omicron,_ = agg_delta.get_features(1, ID_shapvalue)
shap_values_omicron = explainer.shap_values(np.array(features_omicron),
                                          check_additivity=True)

label_omicron = onehot(var)
check_additivity(model, shap_values_omicron, label_omicron, features_omicron, base_value[index])
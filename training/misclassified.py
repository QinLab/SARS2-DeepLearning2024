import argparse
import constants as CONST
from decision_tree.utils_dt import plot_confusion_matrix, plot_metrics
from gene_shap.plots import waterfall
from gene_shap.utils_shap import calculate_base_value, convert_ref_to_onehot_lowercase
import numpy as np
import pandas as pd
import shap
from utils_misclass import get_misclass_seq_index_var
import tensorflow as tf


# Argument Parsing
arg_parser = argparse.ArgumentParser(description="SHAP Value Calculation for Missclassified Sequences")
arg_parser.add_argument("-num", "--num_seq", type=int, default=5,
                        help="Number of sequences for base value (default: 5)")

args = arg_parser.parse_args()
num_seq = args.num_seq

df_train = pd.read_csv(CONST.TRAIN_DIR)
df_test = pd.read_csv(CONST.TEST_DIR)
variants = CONST.VOC_WHO
model = tf.keras.models.load_model(f'{CONST.MODEL_SAVE}')
ref_seq, _, _ = convert_ref_to_onehot_lowercase()


# Get some sequences from test dataset to produce base value
base_features = []
for variant in variants:
    features = calculate_base_value(df_train, variant, num_seq, ID_basevalue=None)
    base_features.append(features)
combined_features = np.concatenate(base_features, axis=0)

# Calculate base value
explainer = shap.DeepExplainer(model, combined_features)
base_value = np.array(explainer.expected_value)

pred_miss, concatenate_feature, y_preds, y_test, misclassified_indices_array = get_misclass_seq_index_var(df_test, model)

plot_confusion_matrix('cnn', y_preds, y_test, 'blue', 'Blues')
plot_metrics('cnn', y_preds, y_test, 'Blues')
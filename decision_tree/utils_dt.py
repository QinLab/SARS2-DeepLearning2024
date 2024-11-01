import constants.constants as CONST
from gene_shap.utils_shap import convert_ref_to_onehot_lowercase
import joblib
from matplotlib import pyplot as plt
import numpy as np
from one_hot.one_hot import *
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import shap
import tensorflow as tf
from tqdm import trange

result_dir = CONST.RSLT_DIR
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def get_data(df):
    
    X = []
    label = []
    y = []

    for i in trange(len(df)):
        X.append(one_hot_encode_seq(df.iloc[i]['sequence']))
        label.append(one_hot_encode_label(df.iloc[i]['Variant_VOC']))
        
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    label = tf.constant(label)
    indices = tf.where(label == 1)
    y = indices[:, 1].numpy()
    
    indices_shuffle = np.random.permutation(len(X))
    X = X[indices_shuffle]
    y = y[indices_shuffle]
    
    return X, y


def get_column_names(seq):

    ref,_,_ = convert_ref_to_onehot_lowercase()
    column_names = []
    chars = ['_', 'A', 'C', 'G', 'I', 'N', 'T']
    for i in range(len(seq)):
        ref_nuc = ref[i]
        for char in chars:
            column_names.append(f'{ref_nuc}{i+1}{char}')

    return column_names


def get_data_shap(df):
    one_hot_seq = []
    one_hot_label = []
    for i in trange(len(df)):
        one_hot_seq.append(one_hot_encode_seq(df['sequence'][i]))
        one_hot_label.append(one_hot_encode_label(df['Variant_VOC'][i]))
    
    return one_hot_seq, one_hot_label


def get_shap_instance(model, df, var):
    
    x, y = get_data(df)
    seq = df[df['Variant_VOC'] == var]['sequence'].values[0]
    column_names = get_column_names(seq)
    df_shap = pd.DataFrame(x, columns=column_names)
    choosen_instance = df_shap.loc[[0]]
    model = joblib.load(model)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(choosen_instance)
    
    return explainer, shap_values, choosen_instance, df_shap
    


def plot_confusion_matrix(model, y_preds, y_test, color, cmap):
    
    conf_matrix = confusion_matrix(y_test, y_preds)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set(font_scale=2)

    heatmap = sns.heatmap(conf_matrix_normalized,
                          annot=True,
                          cmap=cmap,  
                          cbar=False,
                          xticklabels=CONST.VOC_WHO,
                          yticklabels=CONST.VOC_WHO)

    ax.set_xlabel('Predicted Variant', fontsize=26, color=color)
    ax.set_ylabel('True Variants', fontsize=26, color=color)

    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16)  

    plt.title(f'Confusion Matrix for {model}', fontsize=26)
    plt.savefig(f'{result_dir}/conf_matrix/confusion_matrix_{model}.png', dpi=100, bbox_inches='tight')


def plot_metrics(model, y_preds, y_test, palette):

    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='macro')
    recall = recall_score(y_test, y_preds, average='macro')
    f1 = f1_score(y_test, y_preds, average='macro')

    # Create a bar plot for the metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8, 6))
    sns.set(style='whitegrid')

    sns.barplot(x=metrics, y=scores, palette=palette)
    plt.ylim(0, 1) 
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20)
    plt.xlabel('Metrics', fontsize=26)
    plt.ylabel('Scores', fontsize=26)

    # Annotate the bars with the actual values
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.2f}", ha='center', fontsize=26)
        
    plt.savefig(f'{result_dir}/metrics_plot/metrics_{model}.png', dpi=100, bbox_inches='tight')


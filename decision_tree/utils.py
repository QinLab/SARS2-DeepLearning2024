from matplotlib import pyplot as plt
import constants as CONST
import numpy as np
from one_hot import *
from shap.utils import *
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tqdm import trange


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


def plot_confusion_matrix(model, y_preds, y_test, color, cmap):
    
    conf_matrix = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set(font_scale=1.8)
    heatmap = sns.heatmap(conf_matrix,
                          annot=True,
                          fmt='g',
                          cmap=cmap,
                          cbar=False,
                          xticklabels=CONST.VOC_WHO,
                          yticklabels=CONST.VOC_WHO)

    ax.set_xlabel('Predicted Variant', fontsize=26, color=color)
    ax.set_ylabel('True Variants', fontsize=26, color=color)

    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    plt.title('Confusion Matrix for {model}', fontsize=26)
    plt.savefig(f'{CONST.CNFMTRX_DIR}/confusion_matrix_{model}.png', dpi=300, bbox_inches='tight')


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
        
    plt.savefig(f'{CONST.MTC_PLT}/metrics_{model}.png', dpi=300, bbox_inches='tight')

def get_column_names():

    ref,_,_ = convert_ref_to_onehot_lowercase()
    column_names = []
    chars = ['_', 'A', 'C', 'G', 'I', 'N', 'T']
    for i in range(len(seq)):
        ref_nuc = ref[i]
        for char in chars:
            column_names.append(f'{ref_nuc}{i+1}{char}')

    return column_names
from constants import constants as CONST
from gene_shap.utils_agg_shap import Agg_SHAP as aggshap
from gene_shap.plots.waterfallMiss import waterfall
from one_hot.one_hot import one_hot_encode_label
import numpy as np
import pandas as pd

def get_misclass_seq_index_var(df, model):
    
    perdict_stack = []
    feature_list = []
    concatenate_feature_miss=[]
    Ids_stack = []
    labels_array = []
    misclassified_indices_array = []
    pred_miss = []
    variants = CONST.VOC_WHO
    
    for var in variants:  
        df_var = df[df['Variant_VOC'] == var]
        num = len(df_var)
        
        shap_instance = aggshap(df=df_var, var=var)
        _, features_test, IDs = shap_instance.get_features(num_seq=num, ID=None)
        feature_list.append(features_test)
        
        predicted = model.predict(np.array(features_test)) 
        perdict_stack.append(predicted)
        
        labels_test = np.array([one_hot_encode_label(var)] * num)
        labels_array.append(labels_test)
        
        misclassified_indices = np.where(np.argmax(labels_test, axis=1) != np.argmax(predicted, axis=1))
        misclassified_indices_array.append(misclassified_indices)
        
        feature_miss = np.array(features_test)[misclassified_indices]
        concatenate_feature_miss.append(feature_miss)
        
        pred = np.argmax(predicted[misclassified_indices])
        pred_miss.append(pred)
        
        ids = [IDs[index] for index in misclassified_indices[0]]
        Ids_stack.append(ids)
     
    concatenate_feature_miss = np.concatenate(concatenate_feature_miss,axis=0)
    y_preds = np.concatenate(perdict_stack,axis=0)
    y_test = np.concatenate(labels_array, axis=0)
    Ids_stack = np.concatenate(Ids_stack,axis = 0)
    misclassified_indices = np.where(np.argmax(y_preds, axis=1)!=np.argmax(y_test, axis=1))
        
    return pred_miss, feature_list, concatenate_feature_miss, y_preds, y_test, misclassified_indices, Ids_stack


def orf_column_names(shuffled_lists, ref_seq):
    df_test = pd.DataFrame({'Value': shuffled_lists}).T
    df_ORFs = pd.read_csv(CONST.ORf_DIR)

    # Function to check if index is in any ORF range and return the ORF name(s)
    def get_orf_names(index, df_orfs):
        orf_names = [] 
        value = shuffled_lists[index-1]
        nuc = get_nuc(value)
        for _, row in df_orfs.iterrows():
            if row['Start'] <= index <= row['End']:
                orf_names.append(f'{ref_seq[index-1]}{index}{nuc}, '+row['Gene'])
        if len(orf_names) == 0:
            orf_names.append(f'{ref_seq[index-1]}{index}{nuc}, None-ORF')
            
        return ', '.join(orf_names)
    
    def get_nuc(value):
        if value == [1, 0, 0, 0, 0, 0, 0]:
            nuc = '_'
        if value == [0, 1, 0, 0, 0, 0, 0]:
            nuc = 'A'
        if value == [0, 0, 1, 0, 0, 0, 0]:
            nuc = 'C'
        if value == [0, 0, 0, 1, 0, 0, 0]:
            nuc = 'G'
        if value == [0, 0, 0, 0, 1, 0, 0]:
            nuc = 'I'
        if value == [0, 0, 0, 0, 0, 1, 0]:
            nuc = 'N'
        if value == [0, 0, 0, 0, 0, 0, 1]:
            nuc = 'T'
        return nuc

    new_column_names = []
    for index in range(1, len(shuffled_lists) + 1):
        new_column_names.append(get_orf_names(index, df_ORFs))

    df_test.columns = new_column_names

    return df_test

    
def plot_id_missclassified(y_preds, y_test, concatenate_feature_miss, misclassified_indices, ref_seq, ids_stack, base_value, explainer):
    my_dict = {}
    variants = CONST.VOC_WHO
    for index, number in enumerate(list(misclassified_indices[0])):
        my_dict[index] = number
        
    for key,value in my_dict.items():
        true_var = int(value/8000)
        label_true = np.argmax(y_test[value])
        label_predicted = np.argmax(y_preds[value])
        
        feature_miss = concatenate_feature_miss[key].tolist()
        id_miss = ids_stack[key]
        print(id_miss)
        mut_orf_miss = orf_column_names(feature_miss, ref_seq)
        shap_values = explainer.shap_values(np.array(feature_miss).reshape(1, 29891, 7), check_additivity=False)
        print(f"True lable:{variants[true_var]}")
        print(f"Predicted lable:{variants[label_predicted]}")
        waterfall(variants[true_var], variants[true_var],
          id_miss,
          mut_orf_miss,
          shap_values[label_true][0],
          base_value[label_true],
                 )

        waterfall(variants[true_var], variants[label_predicted],
          id_miss,
          mut_orf_miss,
          shap_values[label_predicted][0],
          base_value[label_predicted],
                 )

import constants as CONST
from gene_shap.utils_agg_shap import Agg_SHAP as aggshap
from one_hot import one_hot_encode_label
import numpy as np


def get_misclass_seq_index_var(df, model):
    
    perdict_stack = []
    concatenate_feature=[]
    Ids_stack = []
    labels_array = []
    misclassified_indices_array = []
    pred_miss = []
    variants = CONST.VOC_WHO
    i=0
    for var in variants:  
        i+=1
        df_var = df[df['Variant_VOC'] == var]
        num = len(df_var)
        print("num:", num)
        
        shap_instance = aggshap(df=df_var, var=var)
        _, features_test, IDs = shap_instance.get_features(num_seq=num, ID=None)
        
        predicted = model.predict(np.array(features_test)) 
        perdict_stack.append(predicted)
        
        labels_test = np.array([one_hot_encode_label(var)] * num)
        labels_array.append(labels_test)
        
        misclassified_indices = np.where(np.argmax(labels_test, axis=1) != np.argmax(predicted, axis=1))
        misclassified_indices_array.append(misclassified_indices)
        
        feature = np.array(features_test)[misclassified_indices]
        concatenate_feature.append(feature)
        
        pred = np.argmax(predicted[misclassified_indices])
        pred_miss.append(pred)
        
        ids = [IDs[index] for index in misclassified_indices[0]]
        Ids_stack.append(ids)
     
    concatenate_feature = np.concatenate(np.array(concatenate_feature),axis=0)
    y_preds = np.concatenate(np.array(perdict_stack),axis=0)
    y_test = np.concatenate(np.array(labels_array), axis=0)
    Ids_stack = np.concatenate(np.array(Ids_stack),axis = 0)
    misclassified_indices_array = np.concatenate(np.array(misclassified_indices_array), axis = 0)
        
    return pred_miss, concatenate_feature, y_preds, y_test, misclassified_indices_array


import constants as CONST
import numpy as np
import pandas as pd
import one_hot.one_hot as OneHot
from tqdm import trange


class Agg_SHAP:    
    def __init__(self, df, var=None, base_value=False):
        self.var = var
        self.df = df
        self.column_names = ['ID','sequence', 'Variant_VOC']
        self.variants_who = CONST.VOC_WHO
        
        if self.var not in self.variants_who:
            warning_message = f"Warning: '{var}' is not in the list of supported variants: {variants_who}."
            raise ValueError(warning_message)
    
    def features_id(self, filtered_df, num_seq):
        features = []
        ids = []
        for i in trange(num_seq):
            x = filtered_df['sequence'].iloc[i]
            features.append(np.array(OneHot.one_hot_encode_seq(x)))
            ids.append(filtered_df['ID'].iloc[i])
        
        return features, ids


    def get_features(self, num_seq, ID):        
        labels = []        
        var = self.var
        if num_seq == None:
            num_seq = self.df['Variant_VOC'].value_counts()[var]
        
        filtered_df = self.df[self.df['Variant_VOC'] == var][:num_seq][self.column_names]
        # when we want to have random sequences from a certain variant
        if ID==None:       
             features, ids = self.features_id(filtered_df, num_seq)

        # When we want to seperate sequences based on some certain IDs
        elif ID!=None: 
            filtered_df = filtered_df[filtered_df['ID'].isin(ID)]
            features, ids = self.features_id(filtered_df, num_seq)
        
        return filtered_df, features, ids
        
            
    def get_non_zero_shap_values(self, explainer, features_test, IDs):
        shap_no_zero = []
        Id_non_zero = []

        num_voc = self.variants_who.index(self.var)

        for i in trange(len(features_test)):
            shap_values = explainer.shap_values(features_test[i:i+1], check_additivity=False)
            # Get the summaiton of SHAP value over the 3 dimension which is the total SHAP value for a position in a genetic sequence
            sum_shap = np.sum(np.abs(shap_values[num_voc]), axis=-1).sum()

            if sum_shap != 0:
                shap_no_zero.append(np.sum(shap_values[num_voc], axis=-1).flatten())
                Id_non_zero.append(IDs[i])

        return shap_no_zero, Id_non_zero


    def get_shap_for_each_nuc(self, shap_no_zero, Id_non_zero):

        split_df = pd.DataFrame(shap_no_zero)

        split_df.columns = split_df.columns.astype(int)

        new_column_names = ['{}'.format(i) for i in range(1, len(split_df.columns)+1)]

        split_df.columns = new_column_names
        split_df['Accession ID'] = Id_non_zero


        return split_df
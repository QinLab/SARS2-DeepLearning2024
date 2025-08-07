import constants.constants as CONST
from gene_shap.utils_shap import calculate_shap_value
import numpy as np
import pandas as pd
import shap


class HeatMapSHAP():
    
    def __init__(self, df_first, model, explainer, base_value, max_display=10):
        self.df = df_first
        self.model = model
        self.explainer = explainer
        self.base_value = base_value
        self.max_display = max_display        
        self.orders = []
        self.shap_values_dict = dict()
        self.variants = CONST.VOC_WHO
        
    def get_order(self, ):
        for var in self.variants:  
            df_var = self.df[self.df['Variant_VOC'] == var]
            index = CONST.VOC_WHO.index(var)
            ID_shapvalue=None
            _, _, shap_values = calculate_shap_value(self.model, self.explainer, self.base_value, var, df_var, ID_shapvalue, index)
            self.shap_values_dict[var] = shap_values
            values = np.sum(shap_values[index][0],axis=-1)
            order = np.argsort(-np.abs(values))[:self.max_display]
            self.orders.append(order)
        self.orders = np.concatenate(self.orders)
        print(f"Number of orders: {len(self.orders)}")
        return self.orders
    
    def map_values(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
        
    def get_values_var(self,):  
        shap_high = dict()
        self.orders = self.get_order()
        for key, shap_value in self.shap_values_dict.items():
            for as_var in self.variants:
                index = CONST.VOC_WHO.index(as_var)
                values = np.sum(shap_value[index][0],axis=-1)
                sval = values[self.orders]
                key_dict = f'{key}_{as_var}'
                print(f"Key: {key_dict}, Sval Shape: {sval.shape}")
                
                if len(sval) != len(self.orders):
                    raise ValueError(
                        f"Mismatch: {key_dict} has {len(sval)} elements, "
                        f"but {len(self.orders)} columns were expected."
                    )
                    
                shap_high[key_dict] = sval
                
        shap_df = pd.DataFrame.from_dict(shap_high, orient='index', columns=self.orders)
        
        if len(self.orders) != shap_df.shape[1]:
            raise ValueError(
                f"DataFrame column mismatch: {shap_df.shape[1]} columns found, "
                f"but {len(self.orders)} columns expected."
            )
            
        mapped_shap_df = shap_df.applymap(self.map_values)
         
        return mapped_shap_df
    
    
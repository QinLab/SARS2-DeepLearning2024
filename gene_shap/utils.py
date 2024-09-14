from sars.shap.agg_shap import Agg_SHAP as aggshap
from Bio import SeqIO
import multiprocessing as mp
import numpy as np
import os
from sars.one_hot.one_hot import one_hot_encode_label as onehot
import pandas as pd
import sars.constants as CONST
from sars.training.data_prep import read_labels
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def read_single_seq(args):
    
    id_seq = pd.DataFrame(columns=['ID', 'sequence'])
    file_path, Id = args
    sequences = SeqIO.parse(file_path, "fasta")        
    for sequence in sequences:    
        id_seq = sequence.id.split('|')
        if id_seq[0]==Id:
            seq = sequence.seq            
            id_seq = pd.DataFrame({'ID':[Id]
                                  ,'sequence':[''.join(seq)]})
            return id_seq


def find_single_label(df):

    id_label_map = read_labels(CONST.LABEL_DIR, CONST.VOC_WHO)

    merged_df = pd.merge(df, id_label_map, on='ID')[['ID',
                                                     'sequence', 
                                                     'Variant_VOC']]
    return merged_df

# Function to calculate base value for a variant
def calculate_base_value(df, variant, num_seq, ID_basevalue):
    shap_instance = aggshap(df=df, var=variant)
    _, features, _ = shap_instance.get_features(num_seq, ID_basevalue)
    del _
    return features


def convert_ref_to_onehot_lowercase():
    home = os.path.expanduser('~')
    with open(f'{home}/sars/NC_045512_2_.txt', 'r') as file:
        lines = file.readlines()
        ref_seq = ''.join(line.strip() for line in lines)

    converted_sequence = ref_seq.lower()

    characters = ['-', 'A', 'C', 'G', 'I', 'N', 'T']
    ref_seq_array = np.array(list(ref_seq))
    df = pd.DataFrame({'sequence': ref_seq_array})

    # Specify the custom order of the categorical variables
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(categories=[characters]), [0])],
                           remainder='passthrough')

    # Fit and transform the data
    encoded_array = ct.fit_transform(df)
    ref_seq_oneHot = encoded_array.toarray()
    
    return ref_seq, ref_seq_oneHot, converted_sequence


def get_pos_local_shap(df_shap):
    
    column_names = ['ID', 'Variant_VOC']
    split_df = df_shap['sequence'].str.split('', expand=True)
    split_df.columns = split_df.columns.astype(int)
    split_df = split_df.iloc[:,1:-1]

    new_column_names = ['{}'.format(i) for i in range(1,len(split_df.columns)+1)]
    # Assign the new column names to the DataFrame
    split_df.columns = new_column_names

    df_new_seq = pd.concat([df_shap[column_names],split_df], axis=1)
    
    return df_new_seq  


def check_additivity(model, shap_values, labels_test, features_test, base_value):
    shap_val_sum = [abs(np.sum(i).round(3)) for i in shap_values]
    f_x = base_value + shap_val_sum

    for i in range(len(features_test)):
        predict = model.predict(np.stack(features_test[i:i+1])).argmax()
        true_label = np.argmax(labels_test)
        explaination_model = np.array(f_x).argmax()

        print(f"model prediction: {predict}, ({(CONST.VOC_WHO)[predict]})")
        print(f"True value: {true_label}, ({(CONST.VOC_WHO)[true_label]})")
        print(f"Explanatory Model Prediction: {explaination_model}, ({(CONST.VOC_WHO)[explaination_model]}) \n")

        if predict != true_label != explaination_model:
            print("This is not a valid sequence")
            
            
# Function to calculate and check SHAP values for a variant
def calculate_shap_value(model, explainer, base_value, var, df_sequences, ID_shapvalue, base_value_index):
   
    shap_instance = aggshap(df=df_sequences, var=var)
    df, features, _ = shap_instance.get_features(1, ID_shapvalue)
    shap_values = explainer.shap_values(np.array(features), check_additivity=True)
    
    label = onehot(var)
    check_additivity(model, shap_values, label, features, base_value[base_value_index])
    
    return df, features, shap_values

def get_pos_nuc(df, var_name, cs, df_ORFs):
    column_names = ['ID', 'Variant_VOC']
    df = get_pos_local_shap(df)
    df_seq = df.iloc[:,2:]
    dictionary = df_seq.to_dict(orient='records')[0]

    # rename the columns
    new_columns = []
    for col, val in dictionary.items():
        col = int(col)
        for index, row in df_ORFs.iterrows():
            i = 0
            if col >= int(row['Start']) and col <= int(row['End']):
                i += 1
                matching_gene = row['Gene']
                break
            if i == 0:
                matching_gene = 'Non_ORF'
        aa_to_find = f"{cs[col-1]}{col}{val[0].upper()}"
        for key in CONST.AMINO_ACID:
            if aa_to_find in key:
                value = CONST.AMINO_ACID[key]
                break
            else:
                value = "_" 
        new_columns.append(f"{aa_to_find}, {value} ({matching_gene})")
    df = df.rename(columns=dict(zip(df.iloc[:, 2:].columns, new_columns))) 
    return df
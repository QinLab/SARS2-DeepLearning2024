from gene_shap.agg_shap import Agg_SHAP as aggshap
from Bio import SeqIO
import constants.constants as CONST
import dcor
from gene_shap.mutation import find_most_frequent_mutations, print_frequent_mutations
import itertools
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from one_hot.one_hot import one_hot_encode_label as onehot
import pandas as pd
import re
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from training.data_prep import read_labels


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

def get_pos_nuc(df, cs, df_ORFs):
#     column_names = ['ID', 'Variant_VOC']
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


def get_pos_nuc_summation(df, df_ORFs, column_names, features):   
    df_shap = pd.DataFrame(np.array(features).reshape(1, 29891*7),
                           columns=column_names).reset_index(drop=True)
    df = df.reset_index(drop=True)
    
    dictionary = df_shap.to_dict(orient='records')[0]

    # rename the columns
    new_columns = []
    for col, val in dictionary.items():
        col_num = int(re.search(r'\d+', col).group())
        for index, row in df_ORFs.iterrows():
            matching_gene = 'Non_ORF'
            if col_num >= int(row['Start']) and col_num <= int(row['End']):
                matching_gene = row['Gene']
                break               

        for key in CONST.AMINO_ACID:
            value = "_"
            if col in key:
                value = CONST.AMINO_ACID[key]
                break
                 
        new_columns.append(f"{col}, {value} ({matching_gene})")
    df_shap = df_shap.rename(columns=dict(zip(df_shap.columns, new_columns)))
    
    column_names = ['ID', 'Variant_VOC']
    df_sum = pd.concat([df[column_names],df_shap], axis=1)
    return df_sum


def compute_distance_nonlinear_correlation_matrix(merged_df):
    """
    Computes the distance correlation matrix for the variants regarding the mean of their shap values
    """
    names = CONST.VOC_WHO
    df_corr = pd.DataFrame(index=names, columns=names)
    
    # Calculate nonlinear distance correlation for each pair of columns
    for i in names:
        for j in names:
            df_i = np.array(merged_df[i])[:, None]
            df_j = np.array(merged_df[j])[:, None]
            df_corr.at[i, j] = dcor.distance_correlation(df_i, df_j)

    for col in df_corr.columns:
        df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce', downcast='float')

    return df_corr

def plot_correlation(df):
    sns.set(font_scale=1.4)
    cmap = sns.color_palette("Spectral",as_cmap=True)
    cluster_grid = sns.clustermap(df, cmap=cmap,vmin=0, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    directory_path = f"{CONST.CORL_DIR}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    name = f"{directory_path}/corrolation_map.png"
    cluster_grid.fig.savefig(name, dpi=100, bbox_inches='tight')

    
def get_var_shap_count(df_agg, df_orfs, num):

    numeric_df = df_agg[:-1].drop(columns=['Accession ID'])
    column_count = {}
    column_count_new = {}

    for idx, row in numeric_df.iterrows():
        top_shap_columns = row.nlargest(num).index
        for col in top_shap_columns:
            if col in column_count:
                column_count[col] += 1
            else:
                column_count[col] = 1

    for key, value in list(column_count.items()):
        for _, row in df_orfs.iterrows():
            key = int(key)
            matching_gene = 'Non_ORF'
            if key >= int(row['Start']) and key <= int(row['End']):
                matching_gene = row['Gene']
                break

        value = column_count[str(key)]
        key_new = f'{key} ({ matching_gene})'
        column_count_new[key_new] = value

    var_df = pd.DataFrame(list(column_count_new.items()), columns=['Positions', 'Count'])
    var_df = var_df.sort_values(by='Count', ascending=False)
    
    return var_df

def get_var_dna_count(df_train_test, var, freq, df_orfs):
    
    calc_base = aggshap(df_train_test, var)
    df, features, ID = calc_base.get_features( num_seq = None, 
                                     ID = None, 
                                     )
    
    _, _, converted_sequence = convert_ref_to_onehot_lowercase()
    positions, ref_bases, mut_bases, frequency, _ = find_most_frequent_mutations(df, converted_sequence)
    
    mutations = list(zip(positions, ref_bases, mut_bases, frequency))
    sorted_mutations = sorted(mutations, key=lambda x: x[0], reverse=True)

    data = []
    for position, ref_base, mut_base, frequencys in sorted_mutations:
        aa = f'{ref_base.upper()}{position}{mut_base.upper()}'
        matching_gene = 'Non_ORF'
        for index, row in df_orfs.iterrows():
                if position >= int(row['Start']) and position <= int(row['End']):
                    matching_gene = row['Gene']
                    break

        data.append({
            'mutations': f'{aa} ({matching_gene})',
            'frequency': frequencys
        })

    df = pd.DataFrame(data)
    df_var = df[df["frequency"]>=freq].sort_values(by="frequency", ascending=False)
    return df_var


def count_common_positions_all_combinations(all_sets, set_names):
    
    common_sets = {}
    combinations_with_counts = []

    for r in range(2, len(all_sets) + 1):
        for combination in itertools.combinations(enumerate(all_sets), r):
            indexes, sets = zip(*combination)
            common_values = set.intersection(*sets)
            key = ', '.join([set_names[i] for i in indexes])
            common_sets[key] = (common_values, len(common_values))
            combinations_with_counts.append((key, len(common_values)))

    for combination, (values, count) in common_sets.items():
        print(f"Common values between {combination}: {values}, Count: {count}")
    
    return combinations_with_counts


def plot_common_positions_with_rank(all_sets, set_names, top):
    combinations_with_counts = count_common_positions_all_combinations(all_sets, set_names)
    combinations_with_counts.sort(key=lambda x: x[1], reverse=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(25, 15))
    df_combinations = pd.DataFrame(combinations_with_counts, columns=['Combination', 'Count'])

    sns.barplot(x='Count', y='Combination', data=df_combinations, palette='viridis')
    plt.xlabel(f'Count of Common Positions with {top}', fontsize=24)
    plt.ylabel('Combinations of VOCs', fontsize=24)
    plt.title(f'Counts of Common Positions with {top} in Combination of VOCs', fontsize=24)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    for index, value in enumerate(df_combinations['Count']):
        plt.text(value + 0.1, index, str(value), va='center', fontsize=30, color='black')

    yticks_labels = [f"{label} ({i + 1})" for i, label in enumerate(df_combinations['Combination'])]
    plt.yticks(ticks=range(len(yticks_labels)), labels=yticks_labels, fontsize=24)
    
    directory_path = f"{CONST.VEN_DIR}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    plt.savefig(f'{directory_path}/bar_{top}.png', format='png', dpi=40, bbox_inches='tight')

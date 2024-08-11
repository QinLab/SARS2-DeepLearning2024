from __future__ import division

import gc
import pandas as pd
from Bio import SeqIO
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tqdm.autonotebook import tqdm


def read_sequences(path, df_labelsID):
    print("start reading sequence...")
    id_seq = {}
    length_who = len(df_labelsID)

    labels_id_set = set(df_labelsID['ID'])

    sequences = SeqIO.parse(path, "fasta")
    for sequence in tqdm(sequences, desc='Processing sequences'):

        id = sequence.id.split('|')
        if id[0] in labels_id_set:
            id_seq.setdefault(id[0], []).append(sequence.seq)

    id_seq_ = pd.DataFrame({'ID':list(id_seq.keys())
                            , 'sequence':[list(map(str, seqs)) for seqs in id_seq.values()]})
    
    duplicates = id_seq_['sequence'].duplicated()
    id_seq_ = id_seq_[duplicates == False]
    
    # Clean up memory
    del labels_id_set, sequences, id_seq, df_labelsID
    
    # Trigger garbage collection and frees up memory blocks 
    gc.collect()
    
    return id_seq_


def read_labels(path, desired_var):
    print("start reading label...")
    label_who = pd.read_csv(path, sep='\t', usecols=['Accession ID', 'Variant'], dtype={'Variant': str})

    label_who = label_who.dropna(subset=['Variant'])

    desired_var_pattern = '|'.join(desired_var)  # create a pattern like 'var1|var2|var3|...'
    label_who['Variant_VOC'] = label_who['Variant'].str.extract('('+ desired_var_pattern + ')', expand=False)

    label_who = label_who.dropna(subset=['Variant_VOC'])

    label_who = label_who.rename(columns={'Accession ID': 'ID'})
    
    print("unbalanced\n", label_who['Variant_VOC'].value_counts())
    balance_label_who = balanced_data(label_who)
    
    # Clean up memory
    del label_who
    
    # Trigger garbage collection
    gc.collect()
    
    print("length of labels: ", len(balance_label_who))

    return balance_label_who


def balanced_data(df, num_seq = None):
    
    counts_grouped = df.groupby('Variant_VOC')
    
    if num_seq == None:
        count_clade = pd.DataFrame(df['Variant'].value_counts())
        min_count_df = list(count_clade.min())[0]
        #Apply downsampling on each group to have the same number of rows as the minimum count
        id_seq_clade_df = counts_grouped.apply(lambda x: x.sample(min_count_df))
    else:
        id_seq_clade_df = counts_grouped.apply(lambda x: x.sample(num_seq))
    
    # Reset the index of the resulting dataframe
    id_seq_clade_df.reset_index(drop=True, inplace=True)
    print("balanced\n",pd.DataFrame(id_seq_clade_df["Variant_VOC"].value_counts()))
    
    # Clean up memory
    del count_clade, counts_grouped
    
    # Trigger garbage collection
    gc.collect()
    
    return id_seq_clade_df


def map_clade_to_sequence_function(labels_path, sequences_path):
    

    variants_who = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron']
    
    id_label_map = read_labels(labels_path, variants_who)
    id_seq_ds = read_sequences(sequences_path, id_label_map)
    
    
    merged_df = pd.merge(id_seq_ds, id_label_map, on='ID')
    
    # Clean up memory
    del id_seq_ds, id_label_map
    
    # Trigger garbage collection
    gc.collect()
        
    return merged_df


if __name__ == "__main__":
    
    labels_path= "./GISAID/variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv"
    sequences_path="./GISAID/alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa"

    clade_to_sequences_map = map_clade_to_sequence_function(labels_path, sequences_path) 
    clade_to_sequences_map.to_csv(f'./who_dataset.csv')
import sars.constants as CONST
from numpy.random import RandomState
import pandas as pd


def split_data_train_test(data_path = CONST.BALANC_DIR):

    df = pd.read_csv(data_path)
    df['sequence'] = df['sequence'].str.slice(2, -2) #Remove the first 2 and last 2 characters which are "[", two "'", and "]" 
    
    rng = RandomState()
    
    train = df.sample(frac=0.8, random_state=rng)[['ID','sequence', 'Variant_VOC']]
    num_train = train['Variant_VOC'].value_counts()
    print(f'Number of sequences for each variants in training dataset: {num_train}')
    '''
    Delta      35491
    Omicron    34852
    Gamma      33956
    Alpha      33934
    Beta       32897
    '''
    
    test = df.loc[~df.index.isin(train.index)][['ID','sequence', 'Variant_VOC']]
    num_test = test['Variant_VOC'].value_counts()
    print(f'Number of sequences for each variants in testing dataset: {num_test}')
    '''
    Delta      8841
    Omicron    8739
    Gamma      8581
    Alpha      8448
    Beta       8174
    '''
    
    return train, test

def split_data_val(data, ratio):
    
    validation_ratio = ratio   
    x = 1/ratio
    train_ratio = 1 - validation_ratio 
    train_size = int(len(data) * train_ratio)
    validation_size = int(len(data) * validation_ratio)
    train_data = data[:train_size]
    validation_data = data[train_size:train_size + validation_size]
    
    data_split_dict = dict(train = train_data, validation = validation_data)
    
    return data_split_dict

if __name__ == '__main__':
    
    train, test = split_data_train_test()
    train.reset_index().to_csv(CONST.TRAIN_DIR)
    test.reset_index().to_csv(CONST.TEST_DIR)
    
    #check for duplicated across both train and test dataset
    combined_df = pd.concat([train, test], ignore_index=True)
    # Find duplicates in the combined DataFrame
    cross_duplicates = combined_df[combined_df.duplicated(keep=False)]
    print(f"Number of duplicates across training and test datasets: {len(cross_duplicates)}")
    
    '''
    Number of duplicates across training and test datasets: 0
    '''
    
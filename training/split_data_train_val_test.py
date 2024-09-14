import sars.constants as CONST
from numpy.random import RandomState
import pandas as pd


def split_data_train_test(data_path = CONST.BALANC_DIR):

    df = pd.read_csv(data_path)
    df['sequence'] = df['sequence'].str.slice(2, -2) #Remove the first 2 and last 2 characters which are "[", two "'", and "]" 

    train_sample_size = 32000
    test_sample_size = 8000

    rng = RandomState()

    # Create train and test datasets by sampling from each variant
    train = df.groupby('Variant_VOC', group_keys=False).apply(lambda x: x.sample(n=train_sample_size, random_state=rng))
    
    remaining_df = df.loc[~df.index.isin(train.index)]
    test = remaining_df.groupby('Variant_VOC', group_keys=False).apply(lambda x: x.sample(n=test_sample_size, random_state=rng))

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
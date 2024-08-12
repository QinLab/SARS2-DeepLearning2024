from numpy.random import RandomState
import pandas as pd

def split_data_train_test(data_path = './data/who_dataset.csv'):

    df = pd.read_csv(data_path)
    rng = RandomState()

    train = df.sample(frac=0.8, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]
    
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
    train.reset_index().to_csv(f'./data/train/who_train.csv')
    test.reset_index().to_csv(f'./data/test/who_test.csv')
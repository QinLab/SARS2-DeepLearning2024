import numpy as np
from sklearn.compose import ColumnTransformer as CT
from sklearn.preprocessing import OneHotEncoder as OHE


characters = ['-', 'a', 'c', 'g', 'i', 'n', 't']
variants_who = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron'] 


def one_hot_encode_seq(data):    

    '''
    - = [1 0 0 0 0 0 0]
    a = [0 1 0 0 0 0 0]
    c = [0 0 1 0 0 0 0]
    g = [0 0 0 1 0 0 0]
    i = [0 0 0 0 1 0 0]
    n = [0 0 0 0 0 1 0] 
    t = [0 0 0 0 0 0 1]
    '''
    data = list(data)

    # replacing any character not in characters with 'i'
    data = ['i' if char not in characters else char for char in data]
    # Transform list of characters into array
    data_array = np.array(data).reshape(-1,1)

    # Specify the custom order of the categorical variables
    ct = CT(transformers=[('encoder', OHE(categories=[characters]), [0])],
                           remainder='passthrough')

    # Fit and transform the data
    encoded_array = ct.fit_transform(data_array)

     # Convert encoded_array to numpy array
    encoded_array = encoded_array.toarray()

    return encoded_array


def one_hot_encode_label(label):

    '''
    Alpha:  [1, 0, 0, 0, 0]
    Beta: [0, 1, 0, 0, 0]
    Gamma:  [0, 0, 1, 0, 0]
    Delta:  [0, 0, 0, 1, 0]
    Omicron:  [0, 0, 0, 0, 1]
    '''

    one_hot_encoding = {clade: [0]*len(variants_who) for clade in variants_who}
    for i, variant in enumerate(variants_who):
        if label == variant:
            one_hot_encoding[variant][i] = 1
    return one_hot_encoding[label]

import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import sys
from tensorflow.keras.utils import Sequence
import one_hot.one_hot as OneHot


class DataGenerator(Sequence):
    
    def __init__(self, list_IDs, batch_size, dim,
                 n_classes, df, shuffle=True):
        #Initialization
        self.df = df
        self.dim = dim        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle        
        self.characters = ['-', 'a', 'c', 'g', 'i', 'n', 't']
        self.on_epoch_end()

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, self.df)

        return X, y

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __data_generation(self, list_IDs_temp, df):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            df = self.df

            x = df.loc[df['ID'] == list_IDs_temp[i]]['sequence'].tolist()[0]
            X[i,] = OneHot.one_hot_encode_seq(x)

            y = df.loc[df['ID'] == list_IDs_temp[i]]['Variant_VOC'].tolist()[0]
            Y[i]= OneHot.one_hot_encode_label(y)
        
        return X, Y  
    
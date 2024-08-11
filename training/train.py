from __future__ import division
import ast
from Bio import SeqIO
import itertools
from itertools import cycle
import keras_tuner as kt
from keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np
import pylab
import pandas as pd
import random
from scipy import interp
from sklearn import metrics
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D
import time
from tqdm.autonotebook import tqdm



# splitiing data into tran, validation, and test data
def split_data(data, ratio):
    
    validation_ratio = ratio
    
    x = 1/ratio
    
    #test_ratio = (1-ratio)/(x+10)
    train_ratio = 1 - validation_ratio 
    train_size = int(len(data) * train_ratio)
    validation_size = int(len(data) * validation_ratio)
    #test_size = len(data) - train_size - validation_size
    train_data = data[:train_size]
    validation_data = data[train_size:train_size + validation_size]
    #test_data = data[train_size+validation_size:]
    
    data_split_dict = dict(train = train_data, validation = validation_data)
    
    return data_split_dict



class DataGenerator(Sequence):
    
    #'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size, dim,
                 n_classes, df, shuffle=True):
        #Initialization
        self.df = df
        self.count = 0
        self.dim = dim        
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle        
        #self.labels_path = labels_path
        self.characters = ['-', 'a', 'c', 'g', 'i', 'n', 't']
        #self.sequences_path = sequences_path
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

            x = ast.literal_eval(df.loc[df['ID'] == list_IDs_temp[i]]['sequence'].tolist()[0])[0]
            X[i,] = self._one_hot_encode_seq(x)

            y = df.loc[df['ID'] == list_IDs_temp[i]]['Variant_VOC'].tolist()[0]
            Y[i]= self._one_hot_encode_label(y)
        
        return X, Y

    
    
    def _one_hot_encode_seq(self, data):    
        #self.characters = ['-', 'a', 'c', 'g', 'i', 'n', 't'] # onehoting according to this order

        '''
        - = [1 0 0 0 0 0 0]
        a = [0 1 0 0 0 0 0]
        c = [0 0 1 0 0 0 0]
        g = [0 0 0 1 0 0 0]
        i = [0 0 0 0 1 0 0]
        n = [0 0 0 0 0 1 0]
        t = [0 0 0 0 0 0 1]
        '''
        
        characters = ['-', 'a', 'c', 'g', 'i', 'n', 't']

        data = list(data)

        # replacing any character not in characters with 'i'
        data = ['i' if char not in characters else char for char in data]
        # Transform list of characters into array
        data_array = np.array(data).reshape(-1,1)

        # Specify the custom order of the categorical variables
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(categories=[characters]), [0])],
                               remainder='passthrough')

        # Fit and transform the data
        #encoded_array = ct.fit_transform(df)
        encoded_array = ct.fit_transform(data_array)
        
        # Convert encoded_array to numpy array
        encoded_array = encoded_array.toarray()

#         # Find the index of 'n' in characters  #update N = 0
#         n_index = characters.index('n')

#         # Set all rows in the 'n' column to zero
#         encoded_array[:, n_index] = 0
        
        return encoded_array



    def _one_hot_encode_label(self, label):

        '''
        Alpha:  [1, 0, 0, 0, 0]
        Beta: [0, 1, 0, 0, 0]
        Gamma:  [0, 0, 1, 0, 0]
        Delta:  [0, 0, 0, 1, 0]
        Omicron:  [0, 0, 0, 0, 1]
        '''
        #clades_of_interest = ['GRY','GRA','GV','GK','V','O','L','S','G','GR','GH']
        variants_who = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron'] 
        one_hot_encoding = {clade: [0]*len(variants_who) for clade in variants_who}
        for i, variant in enumerate(variants_who):
            if label == variant:
                one_hot_encoding[variant][i] = 1
        return one_hot_encoding[label]
        
if __name__ == '__main__': 

    with tf.device('/GPU:0'):
        print("Simple GPU computation worked!")
        '''
        Number of sequences in each variant rae about:
        Delta      40000
        Omicron    40000
        Gamma      40000
        Alpha      40000
        Beta       40000
        '''

        #datagenerator with dask
        #client = Client(n_workers=4)  #Create a local Dask cluster and connect it to the client

        df = pd.read_csv('/home/qxy699/Data/WHO_representative_random/WHO_training_BetaTest.csv')
        #df = ddf.compute()

        column = df['ID']
        IDs_seq = column.tolist() #list of IDs

        partition_data = split_data(IDs_seq, 0.20)

        labels = df.set_index('ID')['Variant_VOC'].to_dict() #list of labels

        params = {'dim': (29891, 7),
                  'batch_size': 64,
                  'n_classes': 5,
                  'df': df,
                  'shuffle': True}

        training_generator = DataGenerator(partition_data['train'], labels, **params)
        validation_generator = DataGenerator(partition_data['validation'], labels, **params)

        #defining our model
        model = Sequential()
        model.add(Conv1D(filters =196, kernel_size=19, strides=3, activation='relu', 
                         input_shape=(29891,7)))
        model.add(MaxPooling1D(pool_size=5))

        model.add(Conv1D(filters =196, kernel_size=19, strides=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=5))

        model.add(tf.keras.layers.Flatten())

        #Full Connection
        model.add(Dense(units = 164, activation = 'relu'))

        model.add(Dense(units = 42, activation = 'relu'))

        model.add(Dense(units = 20, activation = 'relu'))
        model.add(Dropout(0.40)) # update= more dropout .25 -> .40

        model.add(Dense(5, activation='softmax'))

        opt = Adam(learning_rate=0.0014924)
        model.compile(loss='categorical_crossentropy', optimizer= opt, 
                      metrics=['accuracy'])
        model.summary()



        #start training
        filepath = './weight_acc_loss_Nov_21/WHO_Nov_21_epoch_15.hdf5'


        checkpoint = ModelCheckpoint(filepath=filepath, 
                                     monitor='val_loss',
                                     verbose=1, 
                                     save_best_only=True,
                                     mode='min')

        callbacks = [checkpoint]

        history = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            epochs = 15,
                            callbacks=[callbacks])


        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        plt.savefig("./weight_acc_loss_Nov_21/Training_validation_accuracy_Nov_21_epoch_15.jpg")
        plt.show()

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'])
        plt.savefig("./weight_acc_loss_Nov_21/Training_validation_loss_Nov_21_epoch_15.jpg")
        plt.show()
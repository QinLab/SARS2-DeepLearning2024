import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import Sequence

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
            X[i,] = self._one_hot_encode_seq(x)

            y = df.loc[df['ID'] == list_IDs_temp[i]]['Variant_VOC'].tolist()[0]
            Y[i]= self._one_hot_encode_label(y)
        
        return X, Y

    
    
    def _one_hot_encode_seq(self, data):    

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
        encoded_array = ct.fit_transform(data_array)
        
        # Convert encoded_array to numpy array
        encoded_array = encoded_array.toarray()
        
        return encoded_array



    def _one_hot_encode_label(self, label):

        '''
        Alpha:  [1, 0, 0, 0, 0]
        Beta: [0, 1, 0, 0, 0]
        Gamma:  [0, 0, 1, 0, 0]
        Delta:  [0, 0, 0, 1, 0]
        Omicron:  [0, 0, 0, 0, 1]
        '''
        variants_who = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron'] 
        one_hot_encoding = {clade: [0]*len(variants_who) for clade in variants_who}
        for i, variant in enumerate(variants_who):
            if label == variant:
                one_hot_encoding[variant][i] = 1
        return one_hot_encoding[label]
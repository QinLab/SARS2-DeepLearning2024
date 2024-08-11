from __future__ import division
import ast
from Bio import SeqIO
import constants as CONST
from data_generator import DataGenerator
from model import get_model, plotter
import numpy as np
import pandas as pd
from split_data_train_val_test as split_data_val


if __name__ == '__main__': 

    '''
    Number of sequences in each variant are about:
    Delta      40000
    Omicron    40000
    Gamma      40000
    Alpha      40000
    Beta       40000
    '''

    #preparing data
    df = pd.read_csv(CONST.TRAIN_DIR)

    column = df['ID']
    IDs_seq = column.tolist() #list of IDs

    partition_data = split_data_val(IDs_seq, CONST.SPLIT_RATIO)

    labels = df.set_index('ID')['Variant_VOC'].to_dict() #list of labels

    params = {'dim': (CONST.SEQ_SIZE, CONST.LEN_SIZE),
              'batch_size': 64,
              'n_classes': 5,
              'df': df,
              'shuffle': True}

    training_generator = DataGenerator(partition_data['train'], labels, **params)
    validation_generator = DataGenerator(partition_data['validation'], labels, **params)

    #define the structure of our model
    model = get_model()
    model.summary()

    #start training
    checkpoint = ModelCheckpoint(filepath=CONST.MODEL_SAVE, 
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

    history_file = 'history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    plotter(history_file)
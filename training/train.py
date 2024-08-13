from __future__ import division

import argparse
import ast
from Bio import SeqIO
import constants as CONST
from data_generator import DataGenerator
from model import get_model, plotter
import numpy as np
import pandas as pd
import pickle
from split_data_train_val_test import split_data_val
from tensorflow.keras.callbacks import ModelCheckpoint


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("batch_size", type=int, help="batch size")
arg_parser.add_argument("epochs", type=int, help="number of epochs")
arg_parser.add_argument("-v", "--verbose", type=int, default=1, help="enables tracing execution (default: 1)")

args = arg_parser.parse_args()

batch_size = args.batch_size  #64
epochs = args.epochs  #15
verbose = args.verbose # 1 if not provided, or the value passed by the user

print(f"batch_size: {batch_size}")
print(f"epochs: {epochs}")

if __name__ == '__main__': 

    '''
    Number of sequences in each variant:
    Delta      35491
    Omicron    34852
    Gamma      33956
    Alpha      33934
    Beta       32897
    '''

    #preparing data
    df = pd.read_csv(CONST.TRAIN_DIR)

    column = df['ID']
    IDs_seq = column.tolist() #list of IDs

    partition_data = split_data_val(IDs_seq, CONST.SPLIT_RATIO)

    labels = df.set_index('ID')['Variant_VOC'].to_dict() #list of labels

    params = {'dim': (CONST.SEQ_SIZE, CONST.LEN_SIZE),
              'batch_size': batch_size,
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
                                 verbose=verbose, 
                                 save_best_only=True,
                                 mode='min')

    callbacks = [checkpoint]

    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs = epochs,
                        callbacks=[callbacks])

    history_file = CONST.HST_DIR
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    plotter(history_file)
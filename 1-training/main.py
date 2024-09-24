from __future__ import division

import argparse
import ast
from Bio import SeqIO
import constants.constants as CONST
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from training.data_generator import DataGenerator
from training.model import get_model, plotter
from training.split_data_train_val_test import split_data_val



arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-batch_size", "--batch_size", type=int, help="Batch size")
arg_parser.add_argument("-epochs","--epochs", type=int, help="Number of epochs")
arg_parser.add_argument("-v", "--verbose", type=int, default=1, help="Enables tracing execution (default: 1)")
arg_parser.add_argument("-pkl", "--pickle", action='store_true', help="Save result in pickle file or not (default: False)")

args = arg_parser.parse_args()

batch_size = args.batch_size  # 64
epochs = args.epochs  # 15
verbose = args.verbose # 1 if not provided, or the value passed by the user
pickle_file = args.pickle # False if not provided, or True passed by the user

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

    params = {'dim': (CONST.SEQ_SIZE, CONST.LEN_SIZE),
              'batch_size': batch_size,
              'n_classes': 5,
              'df': df,
              'shuffle': True}

    training_generator = DataGenerator(partition_data['train'], **params)
    validation_generator = DataGenerator(partition_data['validation'], **params)

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
    
    if pickle_file == True:
        history_file = CONST.BST_PARAM_DIR
        if not os.path.exists(history_file):
            os.makedirs(history_file)
        with open(history_file, 'wb') as file:
            pickle.dump(history.history, file)
    else:
        history = history

    plotter(history, pickle_file)
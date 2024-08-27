import os

LEN_SIZE = 7
NM_CPU = None
SEQ_SIZE = 29891
SPLIT_RATIO = 0.2

HOME_DIR = os.path.expanduser("~")
VOC_WHO = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron']
BALANC_DIR = f"{HOME_DIR}/sars/training/data/who_dataset.csv"
BATCH_DIR = f"{HOME_DIR}/sars/first_sequence/batched_sequences"
FRST_DIR = f"{HOME_DIR}/sars/first_sequences/first_seq"
HST_DIR = f"{HOME_DIR}/sars/results/history.pickle"
LABEL_DIR = "/scr/qinlab/GISAID/variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv"
MODEL_SAVE = f"{HOME_DIR}/sars/model/model.hdf5"
RSLT_DIR = f"{HOME_DIR}/sars/results/"
SEQ_DIR = "/scr/qinlab/GISAID/alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa"
SHAP_DIR = f"{HOME_DIR}/sars/shap/agg_shap_value"
TRAIN_DIR = f"{HOME_DIR}/sars/training/data/train/who_train.csv"
TEST_DIR = f"{HOME_DIR}/sars/training/data/test/who_test.csv"
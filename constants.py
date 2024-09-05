import os

LEN_SIZE = 7
NM_CPU = None
SEQ_SIZE = 29891
SPLIT_RATIO = 0.2
VOC_WHO = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Omicron']

CHARS = {'-': [1, 0, 0, 0, 0, 0, 0], 
     'a': [0, 1, 0, 0, 0, 0, 0], 
     'c': [0, 0, 1, 0, 0, 0, 0], 
     'g': [0, 0, 0, 1, 0, 0, 0], 
     'i': [0, 0, 0, 0, 1, 0, 0], 
     'n': [0, 0, 0, 0, 0, 1, 0], 
     't': [0, 0, 0, 0, 0, 0, 1]}

# This dictionary will be updated frequently based on new amino acide changes in SARS-CoV-2
AMINO_ACID = {'C913T':'S216S', 'C1059T':'T85I', 'A2480G':'I559V', 'C2558T':'P585S', 'C3037T':'F106F',
              'C3267T':'T1001I', 'G5230T':'K837N','C5388A':'A1708D', 'T6954C':'I2230T', 'C8782T':'S76S',
              'G10097A':'G15S', 'A10323G':'K90R', 'G11083T':'L37F', 'T11288-':'S3675-', 'C11289-':'S3675-',
              'T11290-':'S3675-', 'G11291-':'G3676-', 'G11292-':'G3676-', 'T11293-':'G3676-', 'T11294-':'F3677-',
              'T11295-':'F3677-', 'T11296-':'F3677-', 'C14408T':'P314L', 'C14805T':'Y446Y', 'G15451A':'G662S', 
              'C16466T':'P1000L', 'G17259T': 'E1264D', 'C17744T':'P504L', 'A17858G':'Y541C', 'C18060T':'L7L',
              'A20268G':'L216L', 'C21575T': 'L5F', 'G21600T':'S13I', 'C21614T':'L18F', 'C21618G':'T19R','C21621A':'T20N',
              'C21638T':'P26S', 'C21762T':'A67V', 'C21767-':'H69-', 'A21768-':'H69-', 'T21769-':'H69-', 'G21770T':'V70F',
              'G21770-':'V70-', 'T21770-':'V70-', 'C21770-':'V70-', 'A21801C':'D80A', 'A21801G': 'D80G', 'C21846T':'T95I',
              'G21974T':'D138Y','G21987A':'G142D', 'T21992-':'Y144-', 'A21993-':'Y144-', 'T21994-':'Y144-','G22018T':'W152C',
              'G22022A':'E154K', 'A22028-':'E156-', 'G22028-':'E156-', 'G22029-':'E156-', 'A22030-':'E156-','G22031-':'E156-', 
              'T22032C':'F157S', 'A22034G':'R158G','T22032-':'F157-', 'T22032-':'F157-', 'T22033-':'F157-', 'C22034-':'F157-',
              'G22132T':'R190S', 'A22206G': 'D215G', 'C22227T':'A222V', 'C22281-':'L242-', 'T22282-':'L242-', 'T22283-':'L242-', 
              'G22284-':'A243-', 'C22285-':'A243-', 'T22286-':'A243-', 'T22287-':'L244-', 'T22288-':'L244-', 'A22289-':'L244-',
              'T22291-':'A243-', 'G22320A':'D253G', 'G22335T':'W258L', 'C22713T':'P384L', 'A22812C':'K417T', 'G22813T': 'K417N',
              'T22917G':'L452R', 'T22917A':'L452Q', 'T22917G':'L452R', 'G22992A':'S477N', 'G22992A':'S477N', 'C22995A':'T478K',
              'G23012A':'E484K', 'G23012C':'E484Q', 'T23031C':'F490S', 'T23042C':'S494P', 'A23063T':'N501Y', 'G23108C':'E516Q',
              'C23271A':'A570D', 'A23403G':'D614G', 'C23525T':'H655Y', 'G23593T':'Q677H', 'C23604A':'P681H', 'C23604G':'P681R', 
              'C23664T':'A701V', 'C23709T':'T716I','C23731T':'T723T','C24138A':'T859N', 'T24224C':'F888L', 'G24410A':'D950N', 
              'G24410C':'D950H', 'A24432G':'Q957R', 'T24506G':'S982A','C24642T':'T1027I', 'A24775T':'Q1071H', 'G24914C':'D1118H',
              'G25135T':'K1191N', 'C25469T':'S26L', 'G25563T':'Q57H', 'G26144T':'G251V', 'C26456T':'P71L', 'T26767C':'I82T', 
              'T27638C':'V82A', 'C27752T':'T120I', 'G28048T':'R52I', 'A28111G':'Y73C', 'T28144C':'L84S', 'G28167A':'E92K',
              'G28248-':'D119-', 'A28249-':'D119-', 'T28250-':'D119-', 'T28251-':'F120-', 'T28252-':'F120-', 'C28253-':'F120-', 
              'C28253T':'F120F', ('G28280C', 'A28281T', 'T28282A'):'D3L', 'C28311T':'P13L','A28461G':'D63G', 'G28881T':'R203M',
              'G29402T':'D377Y', 'C28977T':'S235F' }

HOME_DIR = os.path.expanduser("~")
BALANC_DIR = f"{HOME_DIR}/sars/training/data/who_dataset.csv"
BATCH_DIR = f"{HOME_DIR}/sars/first_sequences/batched_sequences"
DATA_DT = f"{HOME_DIR}/sars/decision_tree/data/data_dt.csv"
CNFMTRX_DIR = f"{HOME_DIR}/sars/results/conf_matrix"
FRST_DIR = f"{HOME_DIR}/sars/first_sequences/first_seq"
HST_DIR = f"{HOME_DIR}/sars/results/history.pickle"
LABEL_DIR ="/scr/qinlab/GISAID/variant_surveillance_tsv_2022_06_16/variant_surveillance.tsv"
MODEL_CTB = f"{HOME_DIR}/sars/decision_tree/model/ctb.joblib"
MODEL_RF = f"{HOME_DIR}/sars/decision_tree/model/random_forest.joblib"
MODEL_SAVE = f"{HOME_DIR}/sars/model/WHO_Apr_4_epoch_15.hdf5"
MODEL_XGB = f"{HOME_DIR}/sars/decision_tree/model/xgb.joblib"
MTC_PLT = f"{HOME_DIR}/sars/results/metrics_plot"
ORf_DIR = f"{HOME_DIR}/sars/orf_csv/ORF.csv"
RSLT_DIR = f"{HOME_DIR}/sars/results/"
SCAT_DIR = f"{HOME_DIR}/sars/results/scatter_plot"
SEQ_DIR = "/scr/qinlab/GISAID/alignment-and-proteins/msa_2022-06-16/2022-06-16_unmasked.fa"
SHAP_DIR = f"{HOME_DIR}/sars/shap/agg_shap_value"
TRAIN_DIR = f"{HOME_DIR}/sars/training/data/train/who_train.csv"
TEST_DIR = f"{HOME_DIR}/sars/training/data/test/who_test.csv"
VIZ_DIR = f"{HOME_DIR}/sars/results/viz_plot"
WTFL_DIR = f"{HOME_DIR}/sars/results/waterfall_plot"

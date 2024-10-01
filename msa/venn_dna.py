import argparse
import constants.constants as CONST
from gene_shap.plots.venn5 import venn5, get_labels
from gene_shap.utils_shap import plot_common_positions_with_rank
import pandas as pd

arg_parser = argparse.ArgumentParser(description="Venn Diagram")
arg_parser.add_argument("-freq", "--frequency", type=int,
                        help="Number of top SHAP values")

args = arg_parser.parse_args()
freq = args.frequency

elements = ['A', 'B', 'G', 'D', 'O']
set_names = CONST.VOC_WHO

df_orfs = pd.read_csv(CONST.ORf_DIR)
df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
df_test = pd.read_csv(f'{CONST.TEST_DIR}')
df_train_test = pd.concat([df_train, df_test])
del df_train, df_test

if __name__ == '__main__':    
    all_sets = get_commonset_who_dna(df_train_test, freq)
    labels = get_labels(all_sets, elements=elements, fill=['number', 'logic'])
    
    #Venn Diagram
    venn5(labels, 'DNA', names=CONST.VOC_WHO, elements=elements)
    
    #Bar Plot
    plot_common_positions_with_rank(all_sets, set_names, 'Most_Frequent_Mutations')
    
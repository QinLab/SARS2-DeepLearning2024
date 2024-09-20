import argparse
import constants.constants as CONST
from gene_shap.plots.venn5 import venn5, get_labels
from gene_shap.utils import get_var_dna_count, plot_common_positions_with_rank
import pandas as pd

arg_parser = argparse.ArgumentParser(description="Venn Diagram")
arg_parser.add_argument("-freq", "--frequency", type=int,
                        help="Number of top SHAP values")

args = arg_parser.parse_args()
freq = args.frequency

elements = ['A', 'B', 'D', 'G', 'O']
set_names = CONST.VOC_WHO

if __name__ == '__main__':
    df_orfs = pd.read_csv(CONST.ORf_DIR)
    df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
    df_test = pd.read_csv(f'{CONST.TEST_DIR}')
    df_train_test = pd.concat([df_train, df_test])
    del df_train, df_test

    df_alpha = get_var_dna_count(df_train_test, 'Alpha', freq, df_orfs)
    df_beta = get_var_dna_count(df_train_test, 'Beta', freq, df_orfs)
    df_gamma = get_var_dna_count(df_train_test, 'Gamma', freq, df_orfs)
    df_delta = get_var_dna_count(df_train_test, 'Delta', freq, df_orfs)
    df_omicron = get_var_dna_count(df_train_test, 'Omicron', freq, df_orfs)
    
    alpha_set = set(df_alpha['mutations'])
    beta_set = set(df_beta['mutations'])
    gamma_set = set(df_gamma['mutations'])
    delta_set = set(df_delta['mutations'])
    omicron_set = set(df_omicron['mutations'])
    
    all_sets = [alpha_set, beta_set, gamma_set, delta_set, omicron_set]
    labels = get_labels(all_sets, elements=elements, fill=['number', 'logic'])
    
    #Venn Diagram
    venn5(labels, 'DNA', names=CONST.VOC_WHO, elements=elements)
    
    #Bar Plot
    plot_common_positions_with_rank(all_sets, set_names, 'Most_Frequent_Mutations')
    
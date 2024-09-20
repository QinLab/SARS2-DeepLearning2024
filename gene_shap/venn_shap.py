import argparse
import constants.constants as CONST
from gene_shap.plots.venn5 import venn5, get_labels
from gene_shap.utils import get_var_shap_count, plot_common_positions_with_rank
import pandas as pd


arg_parser = argparse.ArgumentParser(description="Venn Diagram")
arg_parser.add_argument("-num", "--num_mut", type=int,
                        help="Number of top SHAP values")

args = arg_parser.parse_args()
num = args.num_mut

elements = ['A', 'B', 'D', 'G', 'O']
set_names = CONST.VOC_WHO


if __name__== '__main__':
    df_orfs = pd.read_csv(CONST.ORf_DIR)
    df_alpha = pd.read_csv(f'{CONST.SHAP_DIR}/agg_Alpha_beeswarm.csv')
    df_beta = pd.read_csv(f'{CONST.SHAP_DIR}/agg_Beta_beeswarm.csv')
    df_gamma = pd.read_csv(f'{CONST.SHAP_DIR}/agg_Gamma_beeswarm.csv')
    df_delta = pd.read_csv(f'{CONST.SHAP_DIR}/agg_Delta_beeswarm.csv')
    df_omicron = pd.read_csv(f'{CONST.SHAP_DIR}/agg_Omicron_beeswarm.csv')

    alpha_venn = get_var_shap_count(df_alpha, df_orfs, num)
    beta_venn = get_var_shap_count(df_beta, df_orfs, num)
    gamma_venn = get_var_shap_count(df_gamma, df_orfs, num)
    delta_venn = get_var_shap_count(df_delta, df_orfs, num)
    omicron_venn = get_var_shap_count(df_omicron, df_orfs, num)

    alpha_set = set(alpha_venn['Positions'])
    beta_set = set(beta_venn['Positions'])
    gamma_set = set(gamma_venn['Positions'])
    delta_set = set(delta_venn['Positions'])
    omicron_set = set(omicron_venn['Positions'])

    all_sets = [alpha_set, beta_set, gamma_set, delta_set, omicron_set]
    labels = get_labels(all_sets, elements=elements, fill=['number', 'logic'])

    #Venn Diagram
    venn5(labels, 'SHAP', names=CONST.VOC_WHO, elements=elements)

    #Bar Plot
    plot_common_positions_with_rank(all_sets, set_names, 'Top SHAP Values')

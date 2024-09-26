import argparse
import constants as CONST
from gene_shap.plots.venn5 import venn5, get_labels
from gene_shap.utils_shap import get_commonset_who_shap, plot_common_positions_with_rank


arg_parser = argparse.ArgumentParser(description="Venn Diagram among VOCs")
arg_parser.add_argument("-num", "--num_mut", type=int,
                        help="Number of top SHAP values")
arg_parser.add_argument("-agg", "--agg_shap_high", action='store_true',
                        help="Venn based on greatest SHAP value over all sequences within a variant")
args = arg_parser.parse_args()
num = args.num_mut
agg = args.agg_shap_high

elements = ['A', 'B', 'D', 'G', 'O']
set_names = CONST.VOC_WHO


if __name__== '__main__':
    all_sets, agg = get_commonset_who_shap(num, agg)
    labels = get_labels(all_sets, elements=elements, fill=['number', 'logic'])

    #Venn Diagram
    venn5(labels, 'SHAP', agg,names=CONST.VOC_WHO, elements=elements)

    #Bar Plot
    plot_common_positions_with_rank(all_sets, set_names, 'Top SHAP Values', agg)

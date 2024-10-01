import argparse
import constants as CONST
from gene_shap.plots.venn5 import venn5, get_labels
from gene_shap.utils_shap import get_commonset_who_shap, plot_common_positions_with_rank


arg_parser = argparse.ArgumentParser(description="Venn Diagram among VOCs")
arg_parser.add_argument("-num", "--num_mut", type=int,
                        help="Number of top normalized SHAP values")
arg_parser.add_argument("-thr", "--threshold", type=float, default=None,
                        help="Threshold for choosing appropriate normalized SHAP value")
arg_parser.add_argument("-perc", "--percentage", type=float, default=None,
                        help="Percentage for choosing top normalized SHAP value")
arg_parser.add_argument("-agg", "--agg_shap_high", action='store_true',
                        help="Venn based on greatest SHAP value over all sequences within a variant")

args = arg_parser.parse_args()
num = args.num_mut
agg = args.agg_shap_high
thr = args.threshold
perc = args.percentage

elements = ['A', 'B', 'G', 'D', 'O']
set_names = CONST.VOC_WHO


if __name__== '__main__':
    all_sets, agg = get_commonset_who_shap(thr, num, perc, agg=True)
    labels = get_labels(all_sets, elements=elements, fill=['number', 'logic'])

    #Venn Diagram
    venn5(labels, f'SHAP{num}', agg,names=CONST.VOC_WHO, elements=elements)

    #Bar Plot
    plot_common_positions_with_rank(all_sets, set_names, f'Top SHAP Values {num}', agg)

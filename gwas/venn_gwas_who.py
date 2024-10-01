import argparse
import constants.constants as CONST
from gwas.utils_gwas import get_commonset_who_pvalue
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os


arg_parser = argparse.ArgumentParser(description="Venn Diagram for GWAS and VOCs")
arg_parser.add_argument("-num", "--num_mut", type=int,
                        help="Number of top normalized(-log(p-values))and normalized SHAP values")
arg_parser.add_argument("-thr", "--threshold", type=float, default=None,
                        help="Threshold for choosing appropriate normalized(-log(p-values)) and normalized SHAP value")
arg_parser.add_argument("-perc", "--percentage", type=float, default=None,
                        help="Percentage for choosing top normalized(-log(p-values)) and normalized SHAP value")
arg_parser.add_argument("-agg", "--agg_shap_high", action='store_true',
                        help="Venn based on greatest SHAP value over all sequences within a variant")
arg_parser.add_argument("-allvar", "--common_all_var", action='store_true',
                        help="Venn based on common greatest SHAP value over all variant")
args = arg_parser.parse_args()
num = args.num_mut
thr = args.threshold
agg = args.agg_shap_high
perc = args.percentage
common_all_var = args.common_all_var


if __name__ == '__main__':
    set_pvalue, set_common_set, agg, name_plot = get_commonset_who_pvalue(agg=agg, common_all_var=common_all_var, num=num, thr=thr, perc=perc)
    venn2(subsets = (set_pvalue, set_common_set), set_labels = ('GWAS', 'SHAP'))
    
    directory_path = f"{CONST.RSLT_DIR}/venn_plot"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    if agg:
        name = 'all'
    else:
        name = 'individual'
        
    plt.savefig(f'{directory_path}/venn_GWAS_VOCs_{name}_{name_plot}.png', format='png', dpi=140, bbox_inches='tight')
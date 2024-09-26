import argparse
import constants.constants as CONST
from gwas.utils_gwas import get_commonset_who_pvalue
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import os


arg_parser = argparse.ArgumentParser(description="Venn Diagram for GWAS and VOCs")
arg_parser.add_argument("-num", "--num_mut", type=int,
                        help="Number of top SHAP values")
arg_parser.add_argument("-thr", "--threshold", type=float,
                        help="Threshold for choosing appropriate normalized(-log(p-values))")
arg_parser.add_argument("-agg", "--agg_shap_high", action='store_true',
                        help="Venn based on greatest SHAP value over all sequences within a variant")
args = arg_parser.parse_args()
num = args.num_mut
thr = args.threshold
agg = args.agg_shap_high


if __name__ == '__main__':
    set_pvalue, set_common_set, agg = get_commonset_who_pvalue(thr=thr, num=num, agg=agg)
    venn2(subsets = (set_pvalue, set_common_set), set_labels = ('GWAS', 'SHAP'))
    
    directory_path = f"{CONST.RSLT_DIR}/venn_plot"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
    if agg:
        name = 'all'
    else:
        name = 'individual'
        
    plt.savefig(f'{directory_path}/venn_GWAS_VOCs_{name}.png', format='png', dpi=140, bbox_inches='tight')
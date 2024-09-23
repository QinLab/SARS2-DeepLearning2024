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

args = arg_parser.parse_args()
num = args.num_mut
thr = args.threshold


if __name__ == '__main__':
    set_pvalue, set_common_set = get_commonset_who_pvalue(thr=thr, num=num)
    venn2(subsets = (set_pvalue, set_common_set), set_labels = ('GWAS', 'WHO'))
    
    directory_path = f"{CONST.VEN_DIR}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(f'{directory_path}/venn_GWAS_VOCs.png', format='png', dpi=140, bbox_inches='tight')
import argparse
import constants.constants as CONST
from gene_shap.agg_shap import Agg_SHAP
from msa.utils_mutations import find_most_frequent_mutations, print_frequent_mutations
from gene_shap.utils_shap import convert_ref_to_onehot_lowercase
import pandas as pd


arg_parser = argparse.ArgumentParser(description="Get the number of most frequent mutaions")
arg_parser.add_argument("-num", "--num_mut", type=int,
                        help="Number of sequences for MSA")
arg_parser.add_argument("-var", "--variant",
                        help="Variant name for SHAP value calculation; (default: None)")
arg_parser.add_argument("-thr", "--threshold", type=int,
                        help="Threshold for frquenceies of a mutation; (default: None)")


args = arg_parser.parse_args()
num_seq = args.num_mut
var = args.variant
thr = int(args.threshold)


if __name__== "__main__":
    ref_seq, ref_seq_oneHot, converted_sequence = convert_ref_to_onehot_lowercase()

    df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
    df_test = pd.read_csv(f'{CONST.TEST_DIR}')
    df_train_test = pd.concat([df_train, df_test])

    calc_base = Agg_SHAP(df_train_test, var)
    df, features, ID = calc_base.get_features( num_seq = num_seq, 
                                         ID = None, 
                                         )

    positions, ref_bases, mut_bases, frequency, _ = find_most_frequent_mutations(df, converted_sequence)

    print_frequent_mutations(positions, ref_bases, mut_bases, frequency, thr, var)
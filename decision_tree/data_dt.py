import argparse
import constants.constants as CONST
import pandas as pd


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-num", "--num_data", type=int, default=2000, help="Number of data for decision tree")

args = arg_parser.parse_args()
num_data = args.num_data

sampled_dfs = []
train = pd.DataFrame()
test = pd.DataFrame()
df = pd.read_csv(CONST.TRAIN_DIR)

if __name__ == '__main__':
    for variant_value in df['Variant_VOC'].unique():
        sampled_df = df[df['Variant_VOC'] == variant_value].sample(n=num_data,
                                                                   random_state=42)
        sampled_dfs.append(sampled_df)

    result_df = pd.concat(sampled_dfs, ignore_index=True)

    for variant, count in result_df['Variant_VOC'].value_counts().items():
        df_var = result_df[result_df['Variant_VOC']==variant]
        df_train = df_var.sample(n = int(count*(1-CONST.SPLIT_RATIO)))
        df_test = df_var.drop(df_train.index)

        train = pd.concat([train, df_train], ignore_index=True)
        test = pd.concat([test, df_test], ignore_index=True)

    train.to_csv(CONST.TRAIN_DT, index=False)
    test.to_csv(CONST.TEST_DT, index=False)
    print("Training and testing dataset for decision trees were saved successfully!")
import argparse
import constants.constants as CONST
from gene_shap.utils_shap import calculate_base_value
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from utils_heatmap import HeatMapSHAP


def main():
    parser = argparse.ArgumentParser(description='Generate SHAP values heatmap with customizable max_display')
    parser.add_argument('--max_display', 
                        type=int, 
                        default=10, 
                        help='Maximum number of features to display in the heatmap (default: 10), for all features use 29891')
    parser.add_argument('--output_dir',
                        type=str,
                        default='results/heatmaps',
                        help='Output directory for saving the heatmap (default: results/heatmaps)')
    parser.add_argument('--output_name',
                        type=str,
                        default='shap_heatmap.png',
                        help='Output filename for the heatmap (default: shap_heatmap.png)')
    parser.add_argument('--figsize_width',
                        type=int,
                        default=20,
                        help='Figure width in inches (default: 20)')
    parser.add_argument('--figsize_height',
                        type=int,
                        default=16,
                        help='Figure height in inches (default: 16)')
    parser.add_argument('--dpi',
                        type=int,
                        default=300,
                        help='DPI for saved image (default: 300)')
    
    args = parser.parse_args()
    
    print(f"Running SHAP heatmap generation with max_display={args.max_display}")
    
    df_train = pd.read_csv(f'{CONST.TRAIN_DIR}')
    first_sequences = pd.read_csv(f'{CONST.FRST_DIR}/first_detected.csv')
    model = tf.keras.models.load_model(f'{CONST.MODEL_SAVE}')
    variants = CONST.VOC_WHO

    base_features = []
    for variant in variants:
        features = calculate_base_value(df_train, variant, num_seq=5, ID_basevalue=None)
        base_features.append(features)
        
    combined_features = np.concatenate(base_features, axis=0)

    explainer = shap.DeepExplainer(model, combined_features)
    base_value = np.array(explainer.expected_value)

    heat_map = HeatMapSHAP(first_sequences, model, explainer, base_value, max_display=args.max_display)
    mapped_shap_df = heat_map.get_values_var()
    
    print(f"Generated SHAP DataFrame with shape: {mapped_shap_df.shape}")

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    max_abs_value = max(abs(mapped_shap_df.values.min()), abs(mapped_shap_df.values.max()))

    print("Original columns:", mapped_shap_df.columns.tolist())

    if mapped_shap_df.columns.duplicated().any():
        print("Warning: Duplicate columns detected. Removing duplicates...")
        mapped_shap_df = mapped_shap_df.loc[:, ~mapped_shap_df.columns.duplicated()]
        print("After removing duplicates:", mapped_shap_df.columns.tolist())

    current_columns = mapped_shap_df.columns.tolist()
    try:
        column_pairs = [(col, int(col)) for col in current_columns]
        sorted_column_pairs = sorted(column_pairs, key=lambda x: x[1])
        sorted_columns = [pair[0] for pair in sorted_column_pairs]
        print("Sorted columns:", sorted_columns)
    except ValueError as e:
        print(f"Could not convert to integers: {e}")
        sorted_columns = sorted(current_columns)
        print("Sorted columns (as strings):", sorted_columns)

    mapped_shap_df_sorted = mapped_shap_df[sorted_columns]

    plt.figure(figsize=(args.figsize_width, args.figsize_height))  

    sns.heatmap(mapped_shap_df_sorted, 
                cmap=cmap, 
                center=0,
                vmin=-max_abs_value, 
                vmax=max_abs_value,
                linewidths=0.5, 
                linecolor='gray',
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'label': 'SHAP Value'})

    current_labels = [int(label.get_text()) for label in plt.gca().get_xticklabels()]
    new_labels = [str(label + 1) for label in current_labels]
    plt.gca().set_xticklabels(new_labels)

    plt.title(f"Heatmap for SHAP Values (max_display={args.max_display})", fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')

    print(f"Heatmap saved to: {output_path}")
    print(f"Final heatmap shape: {mapped_shap_df_sorted.shape}")
    
    # plt.show()


if __name__ == "__main__":
    main()
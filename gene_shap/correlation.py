from gwas_shap.utils_gwas import mean_agg_shap_values
from gene_shap.utils_shap import compute_distance_nonlinear_correlation_matrix, plot_correlation


if __name__ == '__main__':
    merged_df = mean_agg_shap_values(heatmap=True)
    df_corr = compute_distance_nonlinear_correlation_matrix(merged_df)
    plot_correlation(df_corr)
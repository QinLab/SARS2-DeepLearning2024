# GWAS VS SHAP values

## 1. Calculate Chi-Square test
Run:
```
python3 <root_repository>/gwas/chi_square_test.py
```
Results will be saved in `./p_value_csv/p_values_chi_square.csv`

## 2. Visualize Normalized(-log(p-value)) and SHAP Values
Run:
```
python3 <root_repository>/gwas/gwas_vs_shap.py
```
The plot will be saved in `<root_repository>/results/gwas_shap_plot`

## 3. Venn Diagram Between High SHAP Values common in VOCs and High Normalized(-log(p-value))
Run:
```
python3 <root_repository>/gwas/venn_gwas_who.py
```
The plot will be saved in `<root_repository>/results/venn_plot`
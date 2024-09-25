# SHAP value 


## SHAP Value for Single Sequence of Each VOCs

The `single_shap.py` script allows you to calculate SHAP values under three different scenarios:

1. **SHAP value for the initial sequence of each variant:**
   ```bash
   python3 <root_repository>/gene_shap/single_shap.py -init -var <NAME_VARIANT>
2. **SHAP value for a random sequence of a specific variant:**
    ```bash
    python3 <root_repository>/gene_shap/single_shap.py -rand -var <NAME_VARIANT>

3. **SHAP value for a sequence identified by its GISAID ID:**
    ``` bash
    python3 <root_repository>/gene_shap/single_shap.py -ID_shapvalue <GISAID_ID>


All generated plots (Waterfall, DNA, scatter plots) will be saved in:`<root_repository>/results/<PLOT_NAME>_plot/<VARIANT>/`

Replace `<NAME_VARIANT>` with one of the following variant names: Alpha, Beta, Gamma, Delta, or Omicron.


## Overall SHAP for each variant
To compute and save the SHAP values for all sequences within a specific variant, use the following command:
```
python3 <root_repository>/gene_shap/overall_shap.py -var <NAME_VARIANT>
```
The results will be saved in: `<root_repository>/gene_shap/agg_shap_value`

## Correlation Map Between VOCs
Run:
```
python3 <root_repository>/gene_shap/correlation.py
```
The plot will be saved in: `<root_repository>/results/corrolation_map/`

## Venn Diagram Between VOCs
Run:
```
python3 <root_repository>/gene_shap/venn_shap.py -num <Number_of_Top_SHAP_Values>
```
We set `-num` to `1394` because we identified that there are 1,394 normalized values (−log(p-value)) equal to 1.

The plot will be saved in:
`<root_repository>/results/venn_plot`

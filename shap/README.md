# SHAP value 

After preprocessing our sequences with one-hot encoding, the data is transformed into a three-dimensional array: `(NUM_SEQ, 29891, 7)`. This structure represents each nucleotide with 7 distinct features. To simplify the analysis, we aggregate the SHAP values across these 7 features to produce a single SHAP value for each nucleotide. These aggregated values will be saved for future use, such as for plotting and further analysis.

## Local SHAP Value Calculation and Plotting

The `local_shap.py` script allows you to calculate SHAP values under three different scenarios:

1. **SHAP value for the initial sequence of each variant:**
   ```bash
   python3 local_shap.py -init -var <NAME_VARIANT>
2. **SHAP value for a random sequence of a specific variant:**
    ```bash
    python3 local_shap.py -rand -var <NAME_VARIANT>

3. **SHAP value for a sequence identified by its GISAID ID:**
    ``` bash
    python3 local_shap.py -ID_shapvalue <GISAID_ID>


All generated plots (Waterfall, DNA, scatter plots) will be saved in the following directory:
```
<HOME>/sars/results/<PLOT_NAME>_plot/<VARIANT>
```

Replace `<NAME_VARIANT>` with one of the following variant names: Alpha, Beta, Gamma, Delta, or Omicron.

To execute the script with a SLURM job, use: ```bash sbatch local.sh```
Make sure to modify the last line (`python3 ... `) in `local.sh` according to your specific needs.


# Global SHAP for each variant
To compute and save the SHAP values for all sequences within a specific variant, use the following command:
```
python3 global_shap.py -var <NAME_VARIANT>
```
To execute this script via a SLURM job submission:
```
sbatch global.sh
```
Before running, adjust the line containing ``` python3 global_shap.py -var <NAME_VARIANT> ``` to match your requirements.
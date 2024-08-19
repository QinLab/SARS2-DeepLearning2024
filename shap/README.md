# SHAP value 

After preprocessing our sequences with one-hot encoding, the data has three dimensions: (NUM_SEQ, 29891, 7). This means each nucleotide is represented by 7 features. Therefore, we decided to aggregate the SHAP values across these 7 features to produce a single SHAP value for each nucleotide, which we will save for later use (e.g., for plotting).

To achieve this:
1. Create a folder with name: ```agg_shap_value``` to save SHAP values
2. Run the following command:
```
python3 main.py -var <NAME_VARIATION>
```

Instead of <NAME_VARIATION> you should select a name from these names: Alpha, Beta, Gamma, Delta, Omicron
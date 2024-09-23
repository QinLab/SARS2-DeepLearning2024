# Decision Trees

## 1. Preparing data
Since we cannot use the entire training and test datasets for the 2-layer CNN, we selected 2,000 sequences from each VOC, splitting 20% of them for testing. 

To create these datasets, first create a folder named `data`, then run:
```
python3 ./decision_tree/data_dt.py
```
These subsets are saved in the directories specified by `./decision_tree/data/data_dt_train.csv` for training data and `./decision_tree/data/data_dt_test.csv` for test data.

\*\ If you want to select a different number of sequences from each VOCs, you can use the `-num` tag as shown below:
```
python3 ./decision_tree/data_dt.py -num <Your_Preferred_Number>
```
the defualt value for `num` is 2000.

## 2. Fine Tunning
We separate a specified number of sequences from the created dataset, ensuring it is smaller than the training dataset. By default, 500 sequences from each VOC are selected. If you want to choose a different number, use the following command:
```
python3 ./decision_tree/fine_tunning.py -num <Your_Preferred_Number>
```
the result will be saved in the directory: `./decision_tree/best_params`

## 3. Training
First create a folder named `model` then for trainig decision trees, run this command line:
```
python3 ./decision_tree/train.py
```
Model will be save in this directories `decision_tree/model/`.

Confusion matrix and metrics plot will be saved in these directory respectively:
* `results/conf_matix`
* `results/metrics_plot`

## 4. Force plot for SHAP values
Run the following command:
```
python3 ./decision_tree/shap_dt.py -model <Model_Name>
```
The `Model_Name` can be chosen from the following options:

* `rf` for Random Forest
* `xgb` for XGBoost
* `cat` for CatBoost

Results will be saved in this directory: `results/force_plot/`

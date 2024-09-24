import os
import constants as CONST
from catboost import CatBoostClassifier
from decision_tree.utils_dt import *
import joblib
import json
import pandas as pd
from tqdm import trange
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# Loading data
df_train = pd.read_csv(CONST.TRAIN_DT)
df_test = pd.read_csv(CONST.TEST_DT)

# Prepare data
print("Loading training dataset...")
X_train, y_train = get_data(df_train)
print("Loading testing dataset...")
X_test, y_test = get_data(df_test)

param_dir = CONST.BST_PARAM_DIR

if __name__ == "__main__":
    # ------------------------Random forest------------------------
    with open(f"{param_dir}/best_params_rf.json", "r") as f:
        best_param_rf = json.load(f)
    model = RandomForestClassifier(**best_param_rf)
    model.fit(X_train, y_train)
    joblib.dump(model, CONST.MODEL_RF, compress=3)

    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    print("Accuracy in random forest: %.2f%%" % (accuracy * 100.0))

    plot_confusion_matrix('rf', y_preds, y_test, 'red', 'Reds' )
    plot_metrics('rf', y_preds, y_test, 'Reds')

    # ------------------------XGBoost------------------------
    with open(f"{param_dir}/best_params_xgb.json", "r") as f:
        best_params_xgb = json.load(f)
    model = XGBClassifier(**best_params_xgb)
    model.fit(X_train, y_train)
    joblib.dump(model, CONST.MODEL_XGB, compress=3)

    y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy in XGBoost: %.2f%%" % (accuracy * 100.0))

    plot_confusion_matrix('xgb', y_preds, y_test, 'green', 'Greens' )
    plot_metrics('xgb', y_preds, y_test, 'Greens')

    # ------------------------CatBoost------------------------
    X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    with open(f"{param_dir}/best_params_cat.json", "r") as f:
        best_params_cat = json.load(f)
    model = CatBoostClassifier(**best_params_cat)
    model.fit(X_train_cat, y_train_cat, 
            eval_set=(X_val_cat, y_val_cat),
            verbose=False
    )
    joblib.dump(model, CONST.MODEL_CTB, compress=3)

    y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy in CatBoost: %.2f%%" % (accuracy * 100.0))

    plot_confusion_matrix('cat', y_preds, y_test, 'orange', 'Oranges' )
    plot_metrics('cat', y_preds, y_test, 'Oranges')

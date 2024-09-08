from catboost import CatBoostClassifier
import joblib
import pandas as pd
from tqdm import trange
import sars.constants as CONST
from sars.decision_tree.utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# Loading data
df_train = pd.read_csv(CONST.TRAIN_DT)
df_test = pd.read_csv(CONST.TEST_DT)

# Prepare data
X_train, y_train = get_data(df_train)
X_test, y_test = get_data(df_train)

# Random forest
model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X_train, y_train)
joblib.dump(model, CONST.MODEL_RF, compress=3)

y_preds = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy in random forest: %.2f%%" % (accuracy * 100.0))

plot_confusion_matrix('rf', y_preds, y_test, 'red', 'Reds' )
plot_metrics('rf', y_preds, y_test, color, 'Reds')

# XGBoost
model = XGBClassifier()
model.fit(x_train, y_train)
joblib.dump(model, CONST.MODEL_XGB, compress=3)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy in XGBoost: %.2f%%" % (accuracy * 100.0))

plot_confusion_matrix('xgb', y_preds, y_test, 'green', 'Greens' )
plot_metrics('xgb', y_preds, y_test, color, 'Greens')

# CatBoost
X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
cat_features = list(range(0, X_train.shape[1]))

model = CatBoostClassifier()
model.fit(X_train_cat, y_train_cat, 
        cat_features=cat_features, 
        eval_set=(X_val_cat, y_val_cat),
        verbose=False
)
joblib.dump(model, CONST.MODEL_CTB, compress=3)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy in CatBoost: %.2f%%" % (accuracy * 100.0))

plot_confusion_matrix('cat', y_preds, y_test, 'orange', 'Oranges' )
plot_metrics('cat', y_preds, y_test, color, 'Oranges')

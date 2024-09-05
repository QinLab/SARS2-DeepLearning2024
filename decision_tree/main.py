from catboost import CatBoostClassifier
import joblib
import pandas as pd
import pandas as pd
from tqdm import trange
import sars.constants as CONST
from sars.decision_tree.utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from xgboost import XGBClassifier


X_train = []
label_train = []
y_train = []

X_test = []
label_test = []
y_test = []


# Loading data
df_train = pd.read_csv("/home/phatami/sars/training/data/train/who_train.csv")
df_test = pd.read_csv("/home/phatami/sars/training/data/test/who_test.csv")

# Prepare data
for i in trange(len(df_train)):
    X_train.append(one_hot_encode_seq(df_train['sequence'][i]))
    label_train.append(one_hot_encode_label(df_train['Variant_VOC'][i]))
    
for i in trange(len(df_test)):
    X_test.append(one_hot_encode_seq(df_test['sequence'][i]))
    label_test.append(one_hot_encode_label(df_test['Variant_VOC'][i]))


X_train = np.array(X_train)
label_train = tf.constant(label_train)
indices = tf.where(label_train == 1)
y_train = indices[:, 1].numpy()

X_test = np.array(X_test)
label_test = tf.constant(label_test)
indices = tf.where(label_test == 1)
y_test = indices[:, 1].numpy()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

# Shuffle dataset
indices_train = np.random.permutation(len(X_train))
X_train = X_train[indices_train]
y_train = y_train[indices_train]

indices_test = np.random.permutation(len(X_test))
X_test = X_test[indices_test]
y_test = y_test[indices_test]

# Random forest
model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X_train, y_train)
joblib.dump(model, CONST.MODEL_RF, compress=3)

random_forest_preds = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy in random forest: %.2f%%" % (accuracy * 100.0))

# XGBoost
model = XGBClassifier()
model.fit(x_train, y_train)
joblib.dump(model, CONST.MODEL_XGB, compress=3)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy in XGBoost: %.2f%%" % (accuracy * 100.0))

# CatBoost
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = CatBoostClassifier()
model.fit(X_train, y_train, 
        cat_features=cat_features, 
        eval_set=(X_val, y_val), 
        verbose=False
)
joblib.dump(model, CONST.MODEL_CTB, compress=3)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy in XGBoost: %.2f%%" % (accuracy * 100.0))

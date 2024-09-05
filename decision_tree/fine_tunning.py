import argparse
from catboost import CatBoostClassifier
import json
from sars.decision_tree.utils import *
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost as XGBClassifier



arg_parser.add_argument("-num_fine", "--num_finetunning", type=int, default=1000,
                        help="Number of data for hyperparameter tunning  (default:1000)")
args = arg_parser.parse_args()
num_finetunning = args.num_finetunning


# Loading data
df_train = pd.read_csv("/home/phatami/sars/training/data/train/who_train.csv")
df_test = pd.read_csv("/home/phatami/sars/training/data/test/who_test.csv")

# Selecting a certain number of data for fine tunning
df_train = df_train.groupby('Variant_VOC', group_keys=False).apply(lambda x: x.sample(min(len(x), num_finetunning)))
df_test = df_test.groupby('Variant_VOC', group_keys=False).apply(lambda x: x.sample(min(len(x), num_finetunning)))

# Prepare data
X_train, y_train = get_data(df_train)
X_test, y_test = get_data(df_train)

# ------------------------Random forest------------------------
n_estimators = [int(x) for x in np.linspace(start = 2, stop = 20, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 50, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_model = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf_model, 
                               param_distributions = random_grid, 
                               n_iter = 100, 
                               cv = 3, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs = -1,
                               scoring='accuracy'
                              )

rf_random.fit(X_train, y_train)

best_params_rf = rf_random.best_params_
with open('best_params_rf.json', 'w') as file:
    json.dump(best_params_rf, file)

# ------------------------XGBOOST------------------------
n_estimators = [int(x) for x in np.linspace(start = 2, stop = 20, num = 10)]
max_depth = [int(x) for x in np.linspace(3, 50, num = 15)]
max_depth.append(None)
learning_rate = stats.uniform(0.01, 0.1)
subsample = stats.uniform(0.5, 0.5)

xgb_grid = {
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'subsample': subsample,
    'n_estimators':n_estimators
}

xgb_model = XGBClassifier()
xgb_random = RandomizedSearchCV(xgb_model, 
                                param_distributions=xgb_grid, 
                                n_iter=10, 
                                cv=5,
                                verbose=2,
                                random_state=42,
                                n_jobs = -1
                                scoring='accuracy',
                               )

xgb_random.fit(X_train, y_train)

best_params_xgb = xgb_random.best_params_
with open('best_params_xgb.json', 'w') as file:
    json.dump(best_params_xgb, file)
    
# ------------------------CatBoost------------------------
X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
cat_features = list(range(0, X_train.shape[1]))

cat_model = CatBoostClassifier()

iterations = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
depth = [int(x) for x in np.linspace(start = 2, stop = 16, num = 5)]
learning_rate = stats.uniform(0.01, 0.1)
cat_grid = {
        'iterations': iterations,
        'depth': np.arange(2, 16, 2),
        'learning_rate': learning_rate
    }
    
cat_random = RandomizedSearchCV(cat_model,
                                 param_distributions=cat_grid,
                                 n_iter=10, 
                                 cv=5,
                                 verbose=2,
                                 random_state=42,
                                 n_jobs = -1
                                 scoring='accuracy',
                                 )

cat_random.fit(X_train_cat, y_train_cat, 
                cat_features=cat_features, 
                eval_set=(X_val_cat, y_val_cat),
                verbose=False,
                )

best_params_cat = cat_random.best_params_
with open('best_params_cat.json', 'w') as file:
    json.dump(best_params_cat, file)
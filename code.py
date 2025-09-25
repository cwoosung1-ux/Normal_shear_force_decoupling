
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

df = pd.read_excel('dataset.csv')

df['f7'] = df['Z1'] - df['Z4']
df['f8'] = df['Z2'] - df['Z6']
df['f9'] = df['Z3'] - df['Z5']

input_features  = ['Z1','Z2','Z3','Z4','Z5','Z6','f7','f8','f9']
output_features = ['x','y','Tforce','strain']

X_full = df[input_features]
y_full = df[output_features]
strat_labels = df['force'].astype(str)

X_train, X_tmp, y_train, y_tmp, strat_train, strat_tmp = train_test_split(
    X_full, y_full, strat_labels,
    train_size=0.6, random_state=42, shuffle=True, stratify=strat_labels
)
X_val, X_test, y_val, y_test, strat_val, strat_test = train_test_split(
    X_tmp, y_tmp, strat_tmp,
    test_size=0.5, random_state=42, shuffle=True, stratify=strat_tmp
)

base_estimator = XGBRegressor(
    objective='reg:squarederror',
    n_jobs=-1,
    tree_method='auto',
    random_state=42,
    eval_metric='rmse'
)
multi_xgb = MultiOutputRegressor(base_estimator, n_jobs=-1)

param_grid = {
    'estimator__max_depth': [7],
    'estimator__learning_rate': [0.1],
    'estimator__n_estimators': [1000],
    'estimator__subsample': [0.8],
    'estimator__colsample_bytree': [0.8],
}

grid_search = GridSearchCV(
    estimator=multi_xgb,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0,
    refit=True
)
grid_search.fit(X_train, y_train)

best_params = {k.replace('estimator__', ''): v for k, v in grid_search.best_params_.items()}
base_params = best_params.copy()
base_params.pop('n_estimators', None)

per_target_best_n = {}
for t in output_features:
    xgb_tmp = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        tree_method='auto',
        random_state=42,
        eval_metric='rmse',
        early_stopping_rounds=200,
        n_estimators=max(2000, best_params.get('n_estimators', 1000)),
        **base_params
    )
    xgb_tmp.fit(X_train, y_train[t], eval_set=[(X_val, y_val[t])], verbose=False)
    per_target_best_n[t] = getattr(xgb_tmp, 'best_iteration', xgb_tmp.get_params()['n_estimators'])

X_trval = pd.concat([X_train, X_val], axis=0)
y_trval = pd.concat([y_train, y_val], axis=0)

final_models = {}
for t in output_features:
    final_model = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        tree_method='auto',
        random_state=42,
        eval_metric='rmse',
        n_estimators=per_target_best_n[t],
        **base_params
    )
    final_model.fit(X_trval, y_trval[t], verbose=False)
    final_models[t] = final_model

y_test_pred = np.column_stack([final_models[t].predict(X_test) for t in output_features])
pred_df = pd.DataFrame(y_test_pred, columns=output_features)
pred_df['force'] = df.loc[X_test.index, 'force'].values
pred_df_sorted = pred_df.sort_values(by='force')
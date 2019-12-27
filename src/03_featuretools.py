import logging
import pickle
import numpy as np
import pandas as pd
import featuretools as ft
import lightgbm

from util import read_all, save_df_with_dtypes

format = '%(asctime)s %(message)s'
logging.basicConfig(filename='../logs/03_featuretools.log', level=logging.INFO, format=format)

datas = read_all()
app_train = datas['application_train']
app_test = datas['application_test']
bureau = datas['bureau']
bureau_balance = datas['bureau_balance']
cash = datas['POS_CASH_balance']
previous = datas['previous_application']
installments = datas['installments_payments']
credit = datas['credit_card_balance']

app_test["TARGET"] = np.nan
app = app_train.append(app_test, ignore_index=True, sort=False)

# Entity set with id applications
es = ft.EntitySet(id = 'HomeCredit')

# Entities with a unique index
es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR')
es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV')

# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance, 
                              make_index = True, index = 'bureaubalance_index')
es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash, 
                              make_index = True, index = 'cash_index')
es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                              make_index = True, index = 'installments_index')
es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                              make_index = True, index = 'credit_index')

# Add in the defined relationships
es = es.add_relationships([
    ft.Relationship(es['app']['SK_ID_CURR'],      es['bureau']['SK_ID_CURR']),
    ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU']),
    ft.Relationship(es['app']['SK_ID_CURR'],      es['previous']['SK_ID_CURR']),
    ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV']),
    ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV']),
    ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])
])

agg_primitives = ['sum', 'count', 'min', 'max', 'mean', 'mode']
feature_matrix, feature_names = ft.dfs(
    entityset=es, target_entity='app', agg_primitives=agg_primitives, max_depth=2, features_only=False, verbose=True
)

feature_matrix = feature_matrix.reset_index()
save_df_with_dtypes(feature_matrix, '../data/03_featuretools/feature_matrix.csv')

df_train = feature_matrix.dropna(subset=['TARGET'])
df_test = feature_matrix[feature_matrix['TARGET'].isnull()].drop('TARGET', axis=1)

X_train = df_train.drop('TARGET', axis=1)
y_train = df_train['TARGET']
params = {'objective': 'binary', 'device': 'gpu'}
train_set = lightgbm.Dataset(X_train, y_train)

eval_hist = lightgbm.cv(params, train_set, metrics='auc')
auc = eval_hist['auc-mean'][-1]
logging.info(f'Feature Tools LightGBM TrainSet CV AUC Score:{auc:.5f}')

model = lightgbm.train(params, train_set)
X_test = df_test
y_pred = model.predict(X_test)
df_submission = X_test[['SK_ID_CURR']].copy()
df_submission['TARGET'] = y_pred
df_submission.to_csv('../submission/03_featuretools.csv', index=False)
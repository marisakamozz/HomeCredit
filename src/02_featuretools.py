"""
前処理その２
featuretoolsを用いて集計特徴量を作成する。
"""

import logging
import numpy as np
import featuretools as ft

from util import read_all, dump

def main():
    formatter = '%(asctime)s %(message)s'
    logging.basicConfig(filename='../logs/02_featuretools.log', level=logging.INFO, format=formatter)

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
    entity_set = ft.EntitySet(id='HomeCredit')

    # Entities with a unique index
    entity_set = entity_set.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR')
    entity_set = entity_set.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
    entity_set = entity_set.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV')

    # Entities that do not have a unique index
    entity_set = entity_set.entity_from_dataframe(
        entity_id='bureau_balance', dataframe=bureau_balance, make_index=True, index='bureaubalance_index'
    )
    entity_set = entity_set.entity_from_dataframe(
        entity_id='cash', dataframe=cash, make_index=True, index='cash_index'
    )
    entity_set = entity_set.entity_from_dataframe(
        entity_id='installments', dataframe=installments, make_index=True, index='installments_index'
    )
    entity_set = entity_set.entity_from_dataframe(
        entity_id='credit', dataframe=credit, make_index=True, index='credit_index'
    )

    # Add in the defined relationships
    entity_set = entity_set.add_relationships([
        ft.Relationship(entity_set['app']['SK_ID_CURR'],      entity_set['bureau']['SK_ID_CURR']),
        ft.Relationship(entity_set['bureau']['SK_ID_BUREAU'], entity_set['bureau_balance']['SK_ID_BUREAU']),
        ft.Relationship(entity_set['app']['SK_ID_CURR'],      entity_set['previous']['SK_ID_CURR']),
        ft.Relationship(entity_set['previous']['SK_ID_PREV'], entity_set['cash']['SK_ID_PREV']),
        ft.Relationship(entity_set['previous']['SK_ID_PREV'], entity_set['installments']['SK_ID_PREV']),
        ft.Relationship(entity_set['previous']['SK_ID_PREV'], entity_set['credit']['SK_ID_PREV'])
    ])

    agg_primitives = ['sum', 'count', 'min', 'max', 'mean', 'mode']
    feature_matrix, _ = ft.dfs(
        entityset=entity_set, target_entity='app', agg_primitives=agg_primitives, max_depth=2, features_only=False, verbose=True
    )

    feature_matrix = feature_matrix.reset_index()
    dump(feature_matrix, '../data/02_featuretools/feature_matrix.joblib')

if __name__ == "__main__":
    main()

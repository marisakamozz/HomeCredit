import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from util import read_all, fillna_all

sort_keys = {
    'bureau': 'SK_ID_BUREAU',
    'bureau_balance': ['SK_ID_BUREAU', 'MONTHS_BALANCE'],
    'previous_application': 'SK_ID_PREV',
    'POS_CASH_balance': ['SK_ID_PREV', 'MONTHS_BALANCE'],
    'installments_payments': ['SK_ID_PREV', 'DAYS_INSTALMENT'],
    'credit_card_balance': ['SK_ID_PREV', 'MONTHS_BALANCE']
}

max_len = 50
def expand(a):
    z = np.zeros((max_len - a.shape[0], a.shape[1]), a.dtype)
    return np.concatenate([z, a])

def process(item):
    key, df = item
    print(f'--- {key} ---')
    n_cat_cols = len(df.select_dtypes('category').columns)
    cat = {}
    cont = {}
    for sk_id_curr in tqdm(df_sk_id_curr['SK_ID_CURR']):
        data = df.query('SK_ID_CURR == @sk_id_curr').sort_values(sort_keys[key]).iloc[-max_len:]
        if n_cat_cols > 0:
            cat[sk_id_curr] = expand(data.select_dtypes('category').astype('int').values + 1)
        cont[sk_id_curr] = expand(data.select_dtypes('float32').values)
    if n_cat_cols > 0:
        with open(f'../data/06_sequence/{key}_cat.pkl', mode='wb') as f:
            pickle.dump(cat, f)
    with open(f'../data/06_sequence/{key}_cont.pkl', mode='wb') as f:
        pickle.dump(cont, f)


if __name__ == "__main__":
    all_data = read_all('../data/04_powertransform')
    all_data = fillna_all(all_data)

    app_train = all_data.pop('application_train')
    app_test = all_data.pop('application_test')
    df_sk_id_curr = pd.concat([app_train[['SK_ID_CURR']], app_test[['SK_ID_CURR']]])
    # df_sk_id_curr = df_sk_id_curr.iloc[:100]

    bureau = all_data['bureau']
    bureau_balance = all_data['bureau_balance']
    all_data['bureau_balance'] = pd.merge(bureau, bureau_balance, on='SK_ID_BUREAU')[['SK_ID_CURR', 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']]

    with Pool(6) as p:
        p.map(process, list(all_data.items()))

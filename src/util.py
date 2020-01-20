import os
import pathlib
import pickle
import random
import numpy as np
import pandas as pd
import torch

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def read_file_with_dtypes(file):
    if type(file) is str:
        file = pathlib.Path(file)
    dtypes_file = file.parent / f'dtypes_{file.stem}.pkl'
    if dtypes_file.exists():
        with open(dtypes_file, mode='rb') as f:
            dtypes = pickle.load(f)
        return pd.read_csv(file, dtype=dtypes)
    else:
        return pd.read_csv(file)

def read_all(directory='../data/01_labelencoding'):
    datas = {}
    for file in pathlib.Path(directory).glob('*.csv'):
        datas[file.stem] = read_file_with_dtypes(file)
    return datas

def read_sequences(directory='../data/06_sequence'):
    datas = {}
    for file in pathlib.Path(directory).glob('*.pkl'):
        with open(file, mode='rb') as f:
            datas[file.stem] = pickle.load(f)
    return datas

def save_df_with_dtypes(df: pd.DataFrame, file: str):
    file = pathlib.Path(file)
    df.to_csv(file, index=False)
    dtypes_file = file.parent / f'dtypes_{file.stem}.pkl'
    with open(dtypes_file, mode='wb') as file:
        pickle.dump(df.dtypes.to_dict(), file)

def get_dims(all_data):
    dims = {}
    for key, df in all_data.items():
        key_cols = df.columns[df.columns.str.startswith('SK_ID') + df.columns.str.match('TARGET')].values
        df = df.drop(key_cols, axis=1)
        if len(df.select_dtypes('category').columns) > 0:
            cat_dims = df.select_dtypes('category').nunique()
            emb_dims = cat_dims.apply(lambda x: min(50, (x + 1) // 2))
        else:
            cat_dims = None
            emb_dims = None
        cont_dim = len(df.select_dtypes('float32').columns)
        dims[key] = (cat_dims, emb_dims, cont_dim)
    return dims

def fillna_app(app_train, app_test):
    app = app_train.append(app_test, sort=False)
    for column in app.columns[app.isnull().any()]:
        app[column + '_NAN'] = app[column].isnull().astype(int).astype('category')
    app_train = app.dropna(subset=['TARGET'])
    app_train = app_train.fillna(0)
    app_test = app[app['TARGET'].isnull()].drop('TARGET', axis=1)
    app_test = app_test.fillna(0)
    return app_train, app_test

def fillna_all(all_data):
    all_data['application_train'], all_data['application_test'] = fillna_app(all_data['application_train'], all_data['application_test'])
    for key, df in all_data.items():
        for column in df.columns[df.isnull().any()]:
            df[column + '_NAN'] = df[column].isnull().astype(int).astype('category')
        all_data[key] = df.fillna(0)
    return all_data

class HomeCreditDataset(torch.utils.data.Dataset):
    def __init__(self, app, sequences=None, index=None):
        self.sk_id_curr = app[['SK_ID_CURR']]
        if 'TARGET' in app.columns:
            self.target = app[['TARGET']]
            self.app_cat = app.drop(['SK_ID_CURR', 'TARGET'], axis=1).select_dtypes('category').astype('int')
            self.app_cont = app.drop(['SK_ID_CURR', 'TARGET'], axis=1).select_dtypes('float32')
        else:
            self.target = None
            self.app_cat = app.drop(['SK_ID_CURR'], axis=1).select_dtypes('category').astype('int')
            self.app_cont = app.drop(['SK_ID_CURR'], axis=1).select_dtypes('float32')
        self.sequences = sequences
        self.index = index
        
    def __len__(self):
        if self.index is None:
            return len(self.sk_id_curr)
        else:
            return len(self.index)
        
    def __getitem__(self, idx):
        if self.index is not None:
            idx = self.index[idx]
        sk_id_curr = self.sk_id_curr.iloc[idx].values[0]
        if self.target is not None:
            target = self.target.iloc[idx].values
        if self.sequences is None:
            x = (
                self.app_cat.iloc[idx].values,
                self.app_cont.iloc[idx].values,
            )
        else:
            x = (
                self.app_cat.iloc[idx].values,
                self.app_cont.iloc[idx].values,
                self.sequences['bureau_cat'][sk_id_curr],
                self.sequences['bureau_cont'][sk_id_curr],
                self.sequences['bureau_balance_cat'][sk_id_curr],
                self.sequences['bureau_balance_cont'][sk_id_curr],
                self.sequences['previous_application_cat'][sk_id_curr],
                self.sequences['previous_application_cont'][sk_id_curr],
                self.sequences['POS_CASH_balance_cat'][sk_id_curr],
                self.sequences['POS_CASH_balance_cont'][sk_id_curr],
                self.sequences['installments_payments_cat'][sk_id_curr],
                self.sequences['installments_payments_cont'][sk_id_curr],
                self.sequences['credit_card_balance_cat'][sk_id_curr],
                self.sequences['credit_card_balance_cont'][sk_id_curr],
            )
        if self.target is not None:
            return x, target
        else:
            return sk_id_curr, x

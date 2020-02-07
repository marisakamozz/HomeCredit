"""
各種ユーティリティ関数やクラス
"""

import os
import pathlib
import joblib
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

def dump(data, file):
    joblib.dump(data, file, compress=True)

def read_all(directory='../data/01_labelencoding'):
    datas = {}
    for file in pathlib.Path(directory).glob('*.joblib'):
        datas[file.stem] = joblib.load(file)
    return datas

def read_sequences(directory='../data/06_onehot_seq'):
    datas = {}
    for file in pathlib.Path(directory).glob('*.joblib'):
        datas[file.stem] = joblib.load(file)
    return datas

def get_dims(all_data):
    dims = {}
    for name, df in all_data.items():
        key_cols = df.columns[df.columns.str.startswith('SK_ID') + df.columns.str.match('TARGET')].values
        df = df.drop(key_cols, axis=1)
        if len(df.select_dtypes('category').columns) > 0:
            cat_dims = df.select_dtypes('category').nunique()
            emb_dims = cat_dims.apply(lambda x: min(50, (x + 1) // 2))
        else:
            cat_dims = None
            emb_dims = None
        cont_dim = len(df.select_dtypes(exclude='category').columns)
        dims[name] = (cat_dims, emb_dims, cont_dim)
    return dims

SORT_KEYS = {
    'bureau': 'SK_ID_BUREAU',
    'bureau_balance': ['SK_ID_BUREAU', 'MONTHS_BALANCE'],
    'previous_application': 'SK_ID_PREV',
    'POS_CASH_balance': ['SK_ID_PREV', 'MONTHS_BALANCE'],
    'installments_payments': ['SK_ID_PREV', 'DAYS_INSTALMENT'],
    'credit_card_balance': ['SK_ID_PREV', 'MONTHS_BALANCE']
}

def expand(a, max_len):
    z = np.zeros((max_len - a.shape[0], a.shape[1]), a.dtype)
    return np.concatenate([z, a])


class HomeCreditDataset(torch.utils.data.Dataset):
    def __init__(self, app, sequences=None, index=None):
        self.sk_id_curr = app[['SK_ID_CURR']]
        if 'TARGET' in app.columns:
            self.target = app[['TARGET']]
            self.app_cat = app.drop(['SK_ID_CURR', 'TARGET'], axis=1).select_dtypes('category').astype('int')
            self.app_cont = app.drop(['SK_ID_CURR', 'TARGET'], axis=1).select_dtypes(exclude='category')
        else:
            self.target = None
            self.app_cat = app.drop(['SK_ID_CURR'], axis=1).select_dtypes('category').astype('int')
            self.app_cont = app.drop(['SK_ID_CURR'], axis=1).select_dtypes(exclude='category')
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


class OneHotDataset(torch.utils.data.Dataset):
    def __init__(self, app, sequences=None, index=None):
        self.sk_id_curr = app[['SK_ID_CURR']]
        if 'TARGET' in app.columns:
            self.target = app[['TARGET']]
            self.app = app.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        else:
            self.target = None
            self.app = app.drop(['SK_ID_CURR'], axis=1)
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
            x = self.app.iloc[idx].values
        else:
            x = (
                self.app.iloc[idx].values,
                self.sequences['bureau'][sk_id_curr],
                self.sequences['bureau_balance'][sk_id_curr],
                self.sequences['previous_application'][sk_id_curr],
                self.sequences['POS_CASH_balance'][sk_id_curr],
                self.sequences['installments_payments'][sk_id_curr],
                self.sequences['credit_card_balance'][sk_id_curr],
            )
        if self.target is not None:
            return x, target
        else:
            return sk_id_curr, x


def worker_init_fn(worker_id):
    random.seed(worker_id)

class LoaderMaker:
    def __init__(self, all_data, sequences, args, onehot=False):
        self.all_data = all_data
        self.sequences = sequences
        self.args = args
        self.onehot = onehot
        
    def make(self, index=None, train=True):
        if train:
            app = self.all_data['application_train']
        else:
            app = self.all_data['application_test']
        if self.onehot:
            loader = torch.utils.data.DataLoader(
                OneHotDataset(app, self.sequences, index=index),
                batch_size=self.args.batch_size,
                shuffle=train,
                num_workers=6,
                worker_init_fn=worker_init_fn
            )
        else:
            loader = torch.utils.data.DataLoader(
                HomeCreditDataset(app, self.sequences, index=index),
                batch_size=self.args.batch_size,
                shuffle=train,
                num_workers=6,
                worker_init_fn=worker_init_fn
            )
        return loader

import argparse
import logging
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold

from util import read_all, save_df_with_dtypes, read_file_with_dtypes, LightningModel

parser = argparse.ArgumentParser(description='MLP')
parser.add_argument('--lr', action='store', type=float, default=1e-3)
parser.add_argument('--n_hidden', action='store', type=int, default=256)
parser.add_argument('--batch_size', action='store', type=int, default=200)
args = parser.parse_args()

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app_train = read_file_with_dtypes('../data/04_powertransform/application_train.csv')
app_train = pd.get_dummies(app_train)
app_train = app_train.fillna(app_train.mean())

X_train = app_train.drop(['SK_ID_CURR', 'TARGET'], axis=1).values.astype('float32')
y_train = app_train[['TARGET']].values

n_input = X_train.shape[1]
n_hidden = args.n_hidden

class MLP(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
    def forward(self, x):
        return self.main(x)

def worker_init_fn(worker_id):
    random.seed(worker_id)

def make_dataloader(X, y):
    X = torch.tensor(X)
    y = torch.tensor(y)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )
    return loader

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X_train, y_train)
for train_index, val_index in skf.split(X_train, y_train):
    X, y = X_train[train_index], y_train[train_index]
    X_val, y_val = X_train[val_index], y_train[val_index]
    train_dataloader = make_dataloader(X, y)
    val_dataloader = make_dataloader(X_val, y_val)
    model = LightningModel(
        MLP().to(device),
        nn.BCEWithLogitsLoss(),
        train_dataloader,
        val_dataloader,
        args
    )
    trainer = pl.Trainer(
        default_save_path='../logs',
        gpus=-1,
        max_nb_epochs=10,
        early_stop_callback=None
    )
    trainer.fit(model)
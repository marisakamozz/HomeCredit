"""
MLP (One-Hot Encoding)

メインテーブルのみを利用してMLPで予測を行う。
事前学習＋ファインチューニングのベースライン。
"""

import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold

from util import seed_everything, read_file_with_dtypes, get_dims, OneHotDataset
from plutil import LightningModel, LightningModelNoVal
from model import MLPOneHot

def parse_args():
    parser = argparse.ArgumentParser(description='MLP-OneHot')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=256)
    parser.add_argument('--n_epochs', action='store', type=int, default=10)
    parser.add_argument('--batch_size', action='store', type=int, default=1000)
    return parser.parse_args()

def worker_init_fn(worker_id):
    random.seed(worker_id)

def make_dataloader(application, index, batch_size, train=True):
    loader = torch.utils.data.DataLoader(
        OneHotDataset(application, index=index),
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        worker_init_fn=worker_init_fn
    )
    return loader

def get_logger():
    return TensorBoardLogger('../logs', name='MLP-OneHot')

def main():
    args = parse_args()
    seed_everything(args.seed)

    app_train = read_file_with_dtypes('../data/05_onehot/application_train.csv')
    app_test = read_file_with_dtypes('../data/05_onehot/application_test.csv')

    dims = get_dims({'application_train': app_train})
    _, _, n_input = dims['application_train']
    n_hidden = args.n_hidden

    # CV
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(app_train['SK_ID_CURR'], app_train['TARGET'])
    for train_index, val_index in folds:
        train_dataloader = make_dataloader(app_train, train_index, args.batch_size)
        val_dataloader = make_dataloader(app_train, val_index, args.batch_size)
        model = LightningModel(
            MLPOneHot(n_input, n_hidden),
            nn.BCEWithLogitsLoss(),
            train_dataloader,
            val_dataloader,
            args
        )
        trainer = pl.Trainer(
            default_save_path='../logs',
            gpus=-1,
            max_epochs=args.n_epochs,
            early_stop_callback=None,
            logger=get_logger(),
            row_log_interval=100
        )
        trainer.fit(model)

    # Predict
    train_dataloader = make_dataloader(app_train, None, args.batch_size)
    model = LightningModelNoVal(
        MLPOneHot(n_input, n_hidden),
        nn.BCEWithLogitsLoss(),
        train_dataloader,
        args
    )
    trainer = pl.Trainer(
        default_save_path='../logs',
        gpus=-1,
        max_epochs=args.n_epochs,
        early_stop_callback=None,
        logger=get_logger(),
        row_log_interval=100
    )
    trainer.fit(model)

    test_dataloader = make_dataloader(app_test, None, args.batch_size, train=False)
    device = next(model.parameters()).device
    with torch.no_grad():
        ids = []
        y_pred = []
        for sk_id_curr, x in test_dataloader:
            ids.append(sk_id_curr)
            x = x.to(device)
            y_pred.append(torch.sigmoid(model(x)).cpu())
        ids = torch.cat(ids, dim=0).squeeze().numpy()
        y_pred = torch.cat(y_pred, dim=0).squeeze().numpy()
    df_submission = pd.DataFrame({
        'SK_ID_CURR': ids,
        'TARGET': y_pred
    })
    df_submission.to_csv('../submission/13_mlponehot.csv', index=False)


if __name__ == "__main__":
    main()

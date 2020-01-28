import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.logging import MLFlowLogger
from sklearn.model_selection import StratifiedKFold

from util import seed_everything, read_all, get_dims, read_sequences, fillna_all, HomeCreditDataset
from plutil import LightningModel, LightningModelNoVal
from model import R2N

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=16)
    parser.add_argument('--n_main', action='store', type=int, default=32)
    parser.add_argument('--n_epochs', action='store', type=int, default=10)
    parser.add_argument('--batch_size', action='store', type=int, default=1000)
    return parser.parse_args()

def worker_init_fn(worker_id):
    random.seed(worker_id)

def make_dataloader(index, train=True):
    if train:
        app = all_data['application_train']
    else:
        app = all_data['application_test']
    loader = torch.utils.data.DataLoader(
        HomeCreditDataset(app, sequences, index=index),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=6,
        worker_init_fn=worker_init_fn
    )
    return loader


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    all_data = read_all(directory='../data/04_powertransform')
    all_data = fillna_all(all_data)
    dims = get_dims(all_data)
    sequences = read_sequences()

    # CV
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(all_data['application_train']['SK_ID_CURR'], all_data['application_train']['TARGET'])
    for i, (train_index, val_index) in enumerate(folds):
        train_dataloader = make_dataloader(train_index)
        val_dataloader = make_dataloader(val_index)
        model = LightningModel(
            R2N(dims, args.n_hidden, args.n_main),
            nn.BCEWithLogitsLoss(),
            train_dataloader,
            val_dataloader,
            args
        )
        logger = MLFlowLogger(
            experiment_name='HomeCredit',
            tracking_uri='../logs/mlruns',
            tags={'mlflow.runName': f'R2N-FOLD:{i+1}'}
        )
        trainer = pl.Trainer(
            default_save_path='../logs',
            gpus=-1,
            max_epochs=args.n_epochs,
            early_stop_callback=None,
            logger=logger,
            row_log_interval=100
        )
        trainer.fit(model)

    # Predict
    train_dataloader = make_dataloader(None)
    model = LightningModelNoVal(
        R2N(dims, args.n_hidden, args.n_main),
        nn.BCEWithLogitsLoss(),
        train_dataloader,
        args
    )
    trainer = pl.Trainer(
        default_save_path='../logs',
        gpus=-1,
        max_epochs=args.n_epochs,
        early_stop_callback=None
    )
    trainer.fit(model)

    test_dataloader = make_dataloader(None, train=False)
    with torch.no_grad():
        model.cpu().eval()
        ids = []
        y_pred = []
        for sk_id_curr, x in test_dataloader:
            ids.append(sk_id_curr)
            y_pred.append(torch.sigmoid(model(x)))
        ids = torch.cat(ids, dim=0).squeeze().numpy()
        y_pred = torch.cat(y_pred, dim=0).squeeze().numpy()
    df_submission = pd.DataFrame({
        'SK_ID_CURR': ids,
        'TARGET': y_pred
    })
    df_submission.to_csv('../submission/07_r2n.csv', index=False)

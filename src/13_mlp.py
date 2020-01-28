import argparse
import random
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold

from util import seed_everything, read_file_with_dtypes, get_dims, fillna_app, HomeCreditDataset
from plutil import LightningModel, LightningModelNoVal
from model import MLP

def parse_args():
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=256)
    parser.add_argument('--n_epochs', action='store', type=int, default=10)
    parser.add_argument('--batch_size', action='store', type=int, default=200)
    return parser.parse_args()

def worker_init_fn(worker_id):
    random.seed(worker_id)

def make_dataloader(application, index, batch_size, train=True):
    loader = torch.utils.data.DataLoader(
        HomeCreditDataset(application, index=index),
        batch_size=batch_size,
        shuffle=train,
        num_workers=6,
        worker_init_fn=worker_init_fn
    )
    return loader

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    app_train = read_file_with_dtypes('../data/04_powertransform/application_train.csv')
    app_test = read_file_with_dtypes('../data/04_powertransform/application_test.csv')
    app_train, app_test = fillna_app(app_train, app_test)

    dims = get_dims({'application_train': app_train})
    cat_dims, emb_dims, cont_dim = dims['application_train']
    n_input = emb_dims.sum() + cont_dim
    n_hidden = args.n_hidden

    # CV
    '''
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(app_train['SK_ID_CURR'], app_train['TARGET'])
    for i, (train_index, val_index) in enumerate(folds):
        train_dataloader = make_dataloader(app_train, train_index, args.batch_size)
        val_dataloader = make_dataloader(app_train, val_index, args.batch_size)
        model = LightningModel(
            MLP(cat_dims, emb_dims, n_input, n_hidden),
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
            row_log_interval=100
        )
        trainer.fit(model)
    '''
    
    # Predict
    train_dataloader = make_dataloader(app_train, None, args.batch_size)
    model = LightningModelNoVal(
        MLP(cat_dims, emb_dims, n_input, n_hidden),
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

    test_dataloader = make_dataloader(app_test, None, args.batch_size, train=False)
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
    df_submission.to_csv('../submission/05_mlp.csv', index=False)

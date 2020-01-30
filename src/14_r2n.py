import argparse
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold

from util import seed_everything, read_all, get_dims, read_sequences, LoaderMaker
from plutil import LightningModel, LightningModelNoVal
from model import R2N

def parse_args():
    parser = argparse.ArgumentParser(description='R2N-LSTM')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=16)
    parser.add_argument('--n_main', action='store', type=int, default=32)
    parser.add_argument('--n_epochs', action='store', type=int, default=10)
    parser.add_argument('--batch_size', action='store', type=int, default=1000)
    return parser.parse_args()

def get_logger():
    return TensorBoardLogger('../logs', name='R2N-LSTM')

def main():
    args = parse_args()
    seed_everything(args.seed)

    all_data = read_all(directory='../data/03_powertransform')
    dims = get_dims(all_data)
    sequences = read_sequences()
    loader_maker = LoaderMaker(all_data, sequences, args)

    # CV
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(all_data['application_train']['SK_ID_CURR'], all_data['application_train']['TARGET'])
    for train_index, val_index in folds:
        train_dataloader = loader_maker.make(train_index)
        val_dataloader = loader_maker.make(val_index)
        model = LightningModel(
            R2N(dims, args.n_hidden, args.n_main),
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
    train_dataloader = loader_maker.make()
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

    test_dataloader = loader_maker.make(index=None, train=False)
    device = next(model.parameters()).device
    with torch.no_grad():
        ids = []
        y_pred = []
        for sk_id_curr, (app_cat, app_cont) in test_dataloader:
            ids.append(sk_id_curr)
            app_cat = app_cat.to(device)
            app_cont = app_cont.to(device)
            x = (app_cat, app_cont)
            y_pred.append(torch.sigmoid(model(x)).cpu())
        ids = torch.cat(ids, dim=0).squeeze().numpy()
        y_pred = torch.cat(y_pred, dim=0).squeeze().numpy()
    df_submission = pd.DataFrame({
        'SK_ID_CURR': ids,
        'TARGET': y_pred
    })
    df_submission.to_csv('../submission/14_r2n-lstm.csv', index=False)


if __name__ == "__main__":
    main()

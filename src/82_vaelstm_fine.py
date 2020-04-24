"""
VAE(LSTM)

VAE (LSTM)を用いて履歴テーブルから教師なし表現学習で事前学習し、ファインチューニングする。
訓練データのOOF以外とテストデータを使って学習し、訓練データのOOFを使ってearly stoppingを行う。
"""

import argparse
import pathlib
import math
import joblib
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from util import seed_everything, worker_init_fn, dump, read_all, LoaderMaker
from plutil import load_model, VAELSTMModule, LightningModel, HomeCreditTrainer, predict
from model import PretrainedR2N


class OneHotSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, app, sequence, index=None, app_test=None):
        self.sk_id_curr = app[['SK_ID_CURR']]
        self.sk_id_curr_test = None if app_test is None else app_test[['SK_ID_CURR']]
        self.sequence = sequence
        self.index = index
        
    def __len__(self):
        if self.index is None:
            length = len(self.sk_id_curr)
        else:
            length = len(self.index)
        if self.sk_id_curr_test is not None:
            length += len(self.sk_id_curr_test)
        return length
        
    def __getitem__(self, idx):
        if self.index is not None:
            if idx < len(self.index):
                idx = self.index[idx]
                sk_id_curr = self.sk_id_curr.iloc[idx].values[0]
            else:
                idx = idx - len(self.index)
                sk_id_curr = self.sk_id_curr_test.iloc[idx].values[0]
        else:
            if idx < len(self.sk_id_curr):
                sk_id_curr = self.sk_id_curr.iloc[idx].values[0]
            else:
                idx = idx - len(self.sk_id_curr)
                sk_id_curr = self.sk_id_curr_test.iloc[idx].values[0]
        return self.sequence[sk_id_curr]


def parse_args():
    parser = argparse.ArgumentParser(description='VAE(LSTM) OOF FineTuning')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=16)
    parser.add_argument('--n_main', action='store', type=int, default=32)
    parser.add_argument('--n_epochs', action='store', type=int, default=30)
    parser.add_argument('--patience', action='store', type=int, default=5)
    parser.add_argument('--batch_size', action='store', type=int, default=1000)
    return parser.parse_args()

def pretrain(app_train, app_test, sequences, dims, train_index, val_index, args):
    encoders = {}
    for name, diminfo in dims.items():
        sequence = sequences[name]
        train_loader = torch.utils.data.DataLoader(
            OneHotSequenceDataset(app_train, sequence, index=train_index, app_test=app_test),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        val_loader = torch.utils.data.DataLoader(
            OneHotSequenceDataset(app_train, sequence, index=val_index),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        model = VAELSTMModule(diminfo, args.n_hidden, train_loader, val_loader, args)
        logdir = '../logs/82_vaelstm_fine'
        path = pathlib.Path(logdir) / name
        if not path.exists():
            path.mkdir(parents=True)
        logger = TensorBoardLogger(logdir, name=name)
        early_stopping = EarlyStopping(
            patience=args.patience,
            monitor='val_loss',
            mode='min'
        )
        filepath = pathlib.Path(logdir) / name / f'version_{logger.version}' / 'checkpoints'
        model_checkpoint = ModelCheckpoint(
            str(filepath),
            monitor='val_loss',
            mode='min'
        )
        trainer = pl.Trainer(
            default_save_path=logdir,
            gpus=-1,
            max_epochs=args.n_epochs,
            early_stop_callback=early_stopping,
            logger=logger,
            row_log_interval=100,
            checkpoint_callback=model_checkpoint
        )
        trainer.fit(model)

        best_model = load_model(model, name, trainer.logger.version, logdir=logdir)
        encoders[name] = best_model.model.encoder
    return encoders

def main():
    args = parse_args()
    seed_everything(args.seed)
    app_train = joblib.load('../data/05_onehot/application_train.joblib')
    app_test = joblib.load('../data/05_onehot/application_test.joblib')
    sequences = read_all('../data/06_onehot_seq/')
    dims = joblib.load('../data/07_dims/dims05.joblib')
    app_dims = {}
    app_dims['application_train'] = dims.pop('application_train')
    app_dims['application_test'] = dims.pop('application_test')

    app_data = {'application_train': app_train, 'application_test': app_test}
    loader_maker = LoaderMaker(app_data, sequences, args, onehot=True)

    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(app_train['SK_ID_CURR'], app_train['TARGET'])
    best_models = []
    for train_index, val_index in folds:
        encoders = pretrain(app_train, app_test, sequences, dims, train_index, val_index, args)
        train_dataloader = loader_maker.make(train_index)
        val_dataloader = loader_maker.make(val_index)
        model = LightningModel(
            PretrainedR2N(app_dims, args.n_hidden, args.n_main, encoders),
            nn.BCEWithLogitsLoss(),
            train_dataloader,
            val_dataloader,
            args
        )
        name = '82_vaelstm_fine'
        trainer = HomeCreditTrainer(name, args.n_epochs, args.patience)
        trainer.fit(model)
        best_model = load_model(model, name, trainer.logger.version)
        best_models.append(best_model)

    # Predict
    test_dataloader = loader_maker.make(index=None, train=False)
    df_submission = predict(best_models, test_dataloader)
    df_submission.to_csv(f'../submission/{name}.csv', index=False)


if __name__ == "__main__":
    main()

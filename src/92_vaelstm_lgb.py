"""
VAE(LSTM)

VAE (LSTM)を用いて履歴テーブルから教師なし表現学習で履歴特徴量を抽出する。
訓練データのOOF以外とテストデータを使って学習し、訓練データのOOFを使ってearly stoppingを行う。
履歴特徴量とメインテーブルからLightGBMで予測を行う。
"""

import argparse
import pathlib
import math
import joblib
from tqdm import tqdm
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import mlflow

from util import seed_everything, worker_init_fn, read_all
from plutil import load_model, VAELSTMModule


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
    parser.add_argument('--objective', action='store', type=str, default='binary')
    parser.add_argument('--n_estimators', action='store', type=int, default=10000)
    parser.add_argument('--early_stopping', action='store', type=int, default=100)
    parser.add_argument('--device', action='store', type=str, default='gpu')
    return parser.parse_args()

def pretrain(app_train, sequences, dims, train_index, val_index, args):
    encoders = {}
    for name, diminfo in dims.items():
        sequence = sequences[name]
        train_loader = torch.utils.data.DataLoader(
            OneHotSequenceDataset(app_train, sequence, index=train_index),
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
        logdir = '../logs/92_vaelstm_lgb'
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

def predict(app, encoders, sequences, args):
    for name, model in encoders.items():
        sequence = sequences[name]
        dataloader = torch.utils.data.DataLoader(
            OneHotSequenceDataset(app, sequence),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            encodings = []
            total = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
            for x in tqdm(dataloader, total=total):
                x = x.to(device)
                encoding = model(x)
                encodings.append(encoding)
        encoding = torch.cat(encodings, dim=0).cpu().numpy()
        sk_id_curr = dataloader.dataset.sk_id_curr['SK_ID_CURR'].values
        columns = [f'{name}_{i}' for i in range(encoding.shape[1])]
        df_encoding = pd.DataFrame(encoding, index=sk_id_curr, columns=columns)
        df_encoding = df_encoding.reset_index().rename(columns={'index': 'SK_ID_CURR'})
        app = app.merge(df_encoding)
    return app

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

    mlflow.set_tracking_uri('../logs/mlruns')
    mlflow.set_experiment('HomeCredit')
    run_name = '92_vaelstm_lgb'
    params = vars(args)
    df_submission = app_test[['SK_ID_CURR']].copy()

    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(app_train['SK_ID_CURR'], app_train['TARGET'])
    for i, (train_index, val_index) in enumerate(folds):
        # Train Encoder
        encoders = pretrain(app_train, sequences, dims, train_index, val_index, args)

        # Train LightGBM Model
        app_encoding_train = predict(app_train, encoders, sequences, args)
        x = app_encoding_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        y = app_encoding_train['TARGET']
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_valid, y_valid = x.iloc[val_index], y.iloc[val_index]
        train_set = lgb.Dataset(x_train, y_train)
        valid_set = lgb.Dataset(x_valid, y_valid)
        model = lgb.train(params, train_set, valid_sets=[valid_set])
        y_pred = model.predict(x_valid)
        auc = roc_auc_score(y_valid, y_pred)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metric('auc', auc)
        
        # Predict
        app_encoding_test = predict(app_test, encoders, sequences, args)
        x_test = app_encoding_test.drop('SK_ID_CURR', axis=1)
        y_pred = model.predict(x_test)
        df_submission[f'pred_{i}'] = y_pred
    df_submission = df_submission.set_index('SK_ID_CURR').mean(axis=1).reset_index()
    df_submission.columns = ['SK_ID_CURR', 'TARGET']
    df_submission.to_csv(f'../submission/{run_name}.csv', index=False)


if __name__ == "__main__":
    main()

"""
VAE(LSTM)

VAE (LSTM)を用いて履歴テーブルから教師なし表現学習で履歴特徴量を抽出する。
訓練データ全体を使って学習し、テストデータを使ってearly stoppingを行う。
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

from util import seed_everything, worker_init_fn, dump, read_all
from plutil import load_model, VAELSTMModule


class OneHotSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, app, sequence):
        self.sk_id_curr = app[['SK_ID_CURR']]
        self.sequence = sequence
        
    def __len__(self):
        return len(self.sk_id_curr)
        
    def __getitem__(self, idx):
        sk_id_curr = self.sk_id_curr.iloc[idx].values[0]
        return self.sequence[sk_id_curr]


def parse_args():
    parser = argparse.ArgumentParser(description='VAE(LSTM)')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=16)
    parser.add_argument('--n_epochs', action='store', type=int, default=30)
    parser.add_argument('--patience', action='store', type=int, default=5)
    parser.add_argument('--batch_size', action='store', type=int, default=1000)
    return parser.parse_args()

def predict(name, model, dataloader):
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
    return df_encoding

def main():
    args = parse_args()
    seed_everything(args.seed)
    app_train = joblib.load('../data/05_onehot/application_train.joblib')
    app_test = joblib.load('../data/05_onehot/application_test.joblib')
    sequences = read_all('../data/06_onehot_seq/')
    dims = joblib.load('../data/07_dims/dims05.joblib')
    dims.pop('application_train')
    dims.pop('application_test')

    for name, diminfo in dims.items():
        sequence = sequences[name]
        train_loader = torch.utils.data.DataLoader(
            OneHotSequenceDataset(app_train, sequence),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        test_loader = torch.utils.data.DataLoader(
            OneHotSequenceDataset(app_test, sequence),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        model = VAELSTMModule(diminfo, args.n_hidden, train_loader, test_loader, args)
        logdir = '../logs/22_vaelstm'
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
        train_loader_no_shuffle = torch.utils.data.DataLoader(
            OneHotSequenceDataset(app_train, sequence),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        df_train = predict(name, best_model, train_loader_no_shuffle)
        df_test = predict(name, best_model, test_loader)
        df_encoding = pd.concat([df_train, df_test])
        dump(df_encoding, f'../data/22_vaelstm/{name}.joblib')


if __name__ == "__main__":
    main()

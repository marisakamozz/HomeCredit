import pathlib
import pickle
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

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

def read_all(directory='../data/02_labelencoding'):
    datas = {}
    for file in pathlib.Path(directory).glob('*.csv'):
        datas[file.stem] = read_file_with_dtypes(file)
    return datas

def save_df_with_dtypes(df: pd.DataFrame, file: str):
    file = pathlib.Path(file)
    df.to_csv(file, index=False)
    dtypes_file = file.parent / f'dtypes_{file.stem}.pkl'
    with open(dtypes_file, mode='wb') as file:
        pickle.dump(df.dtypes.to_dict(), file)

class LightningModel(pl.LightningModule):
    def __init__(self, model, loss, train_loader, val_loader, hparams):
        super().__init__()
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hparams = hparams

    def forward(self, x):
        return self.model.forward(x)
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        return {
            'loss': loss,
            'log': {'train_loss': loss}
        }

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        return {
            'y': y,
            'y_pred': y_pred
        }

    def validation_end(self, outputs):
        y = torch.cat([output['y'] for output in outputs], dim=0)
        y_pred = torch.cat([output['y_pred'] for output in outputs], dim=0)
        loss = self.loss(y_pred, y)
        auc = roc_auc_score(y.cpu().numpy(), y_pred.cpu().numpy())
        return {
            'val_loss': loss,
            'auc': auc,
            'log': {
                'val_loss': loss,
                'auc': auc
            }
        }
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader
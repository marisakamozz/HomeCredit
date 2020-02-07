import pathlib
from tqdm import tqdm
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import roc_auc_score


class LightningModelNoVal(pl.LightningModule):
    def __init__(self, model, loss, train_loader, hparams):
        super().__init__()
        self.model = model
        self.loss = loss
        self.train_loader = train_loader
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
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader


class LightningModel(LightningModelNoVal):
    def __init__(self, model, loss, train_loader, val_loader, hparams):
        super().__init__(model, loss, train_loader, hparams)
        self.val_loader = val_loader

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
            'log': {
                'val_loss': loss,
                'auc': auc
            }
        }

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader


class HomeCreditTrainer(pl.Trainer):
    def __init__(self, name, n_epochs, patience):
        logger = TensorBoardLogger('../logs', name=name)
        early_stopping = EarlyStopping(
            patience=patience,
            monitor='auc',
            mode='max'
        )
        filepath = pathlib.Path('../logs') / name / f'version_{logger.version}' / 'checkpoints'
        model_checkpoint = ModelCheckpoint(
            str(filepath),
            monitor='auc',
            mode='max'
        )
        super().__init__(
            default_save_path='../logs',
            gpus=-1,
            max_epochs=n_epochs,
            early_stop_callback=early_stopping,
            logger=logger,
            row_log_interval=100,
            checkpoint_callback=model_checkpoint
        )


def load_model(model, name, version):
    filepath = pathlib.Path('../logs') / name / f'version_{version}' / 'checkpoints'
    filename = next(iter(filepath.glob('*.ckpt')))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict(best_models, test_dataloader):
    df_preds = []
    with torch.no_grad():
        for model in best_models:
            device = next(model.parameters()).device
            model.eval()
            ids = []
            y_pred = []
            for sk_id_curr, x in tqdm(test_dataloader, total=len(test_dataloader.dataset)):
                ids.append(sk_id_curr)
                if isinstance(x, list):
                    x = [data.to(device) for data in x]
                else:
                    x = x.to(device)
                y_pred.append(torch.sigmoid(model(x)).cpu())
            ids = torch.cat(ids, dim=0).squeeze().numpy()
            y_pred = torch.cat(y_pred, dim=0).squeeze().numpy()
            df_pred = pd.DataFrame({
                'SK_ID_CURR': ids,
                'TARGET': y_pred
            })
            df_preds.append(df_pred)
    df_submission = df_preds[0][['SK_ID_CURR']]
    for df_pred in df_preds:
        df_submission = pd.merge(df_submission, df_pred, on='SK_ID_CURR')
    df_submission = df_submission.set_index('SK_ID_CURR').mean(axis=1).reset_index()
    df_submission.columns = ['SK_ID_CURR', 'TARGET']
    return df_submission

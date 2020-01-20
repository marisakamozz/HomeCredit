import torch
import pytorch_lightning as pl

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
import pathlib
import math
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model import LSTMEncoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator, PretrainedR2N, VAELSTM
from util import LoaderMaker


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


def load_model(model, name, version=None, logdir='../logs'):
    if version is None:
        logdir_path = pathlib.Path(logdir) / name
        version = 0
        for subdir in logdir_path.iterdir():
            if subdir.name.startswith('version_'):
                new_version = int(subdir.name[8:])
                if new_version > version:
                    version = new_version
    filepath = pathlib.Path(logdir) / name / f'version_{version}'  # / 'checkpoints'
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
            total = math.ceil(len(test_dataloader.dataset) / test_dataloader.batch_size)
            for sk_id_curr, x in tqdm(test_dataloader, total=total):
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


# for DIM
class DIMLSTMModule(pl.LightningModule):
    def __init__(self, diminfo, n_hidden, train_loader, val_loader, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = LSTMEncoder(diminfo, n_hidden)
        self.global_discriminator = GlobalDiscriminator(diminfo, n_hidden)
        self.local_discriminator = LocalDiscriminator(diminfo, n_hidden)
        self.prior_descriminator = PriorDiscriminator(n_hidden)
        self.criterion = nn.BCEWithLogitsLoss()
        self.encoding = None
        self.train_loader = train_loader
        self.val_loader = val_loader

    def forward(self, x):
        return self.encoder(x)

    def logits_loss(self, logits, label):
        target = torch.full(logits.size(), label, device=logits.device)
        return self.criterion(logits, target)
    
    def prior_loss_function(self, encoding):
        logits_encoding = self.prior_descriminator(encoding)
        prior = torch.rand_like(encoding)
        logits_prior = self.prior_descriminator(prior)
        return self.logits_loss(logits_encoding, 0) + self.logits_loss(logits_prior, 1)

    def main_loss_function(self, encoding, batch, alpha=1.0, beta=1.0, gamma=0.1):
        cat, cont = batch
        cat_fake = torch.cat((cat[1:], cat[0].unsqueeze(0)), dim=0)
        cont_fake = torch.cat((cont[1:], cont[0].unsqueeze(0)), dim=0)
        batch_fake = (cat_fake, cont_fake)
        # global loss
        Ej = - F.softplus(self.global_discriminator(encoding, batch) * (-1)).mean()
        Em = F.softplus(self.global_discriminator(encoding, batch_fake)).mean()
        global_loss = alpha * (Em - Ej)
        # local loss
        Ej = - F.softplus(self.local_discriminator(encoding, batch) * (-1)).mean()
        Em = F.softplus(self.local_discriminator(encoding, batch_fake)).mean()
        local_loss = beta * (Em - Ej)
        # encoder loss
        logits_encoding = self.prior_descriminator(encoding)
        encoder_loss = gamma * self.logits_loss(logits_encoding, 1)
        return global_loss + local_loss + encoder_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        # train prior discriminator
        if optimizer_idx == 0:
            self.encoding = self.encoder(batch)
            loss = self.prior_loss_function(self.encoding.detach())
            return {
                'loss': loss,
                'log': {'train_loss_prior': loss}
            }
        # train encoder
        if optimizer_idx == 1:
            loss = self.main_loss_function(self.encoding, batch)
            return {
                'loss': loss,
                'log': {'train_loss_main': loss}
            }
    
    def validation_step(self, batch, batch_idx):
        encoding = self.encoder(batch)
        main_loss = self.main_loss_function(encoding, batch).item()
        prior_loss = self.prior_loss_function(encoding).item()
        return {
            'main_loss': main_loss,
            'prior_loss': prior_loss
        }

    def validation_end(self, outputs):
        main_loss = sum([output['main_loss'] for output in outputs]) / len(outputs)
        prior_loss = sum([output['prior_loss'] for output in outputs]) / len(outputs)
        return {
            'log': {
                'val_loss_main': main_loss,
                'val_loss_prior': prior_loss
            }
        }

    def configure_optimizers(self):
        prior_optimizer = torch.optim.Adam(self.prior_descriminator.parameters(), lr=self.hparams.lr)
        main_module = nn.ModuleList([self.encoder, self.global_discriminator, self.local_discriminator])
        main_optimizer = torch.optim.Adam(main_module.parameters(), lr=self.hparams.lr)
        return [prior_optimizer, main_optimizer], []

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader


class VAELSTMModule(pl.LightningModule):
    def __init__(self, diminfo, n_hidden, train_loader, val_loader, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = VAELSTM(diminfo, n_hidden)
        self.criterion = nn.MSELoss(reduction='sum')
        self.train_loader = train_loader
        self.val_loader = val_loader

    def loss_function(self, recon_x, x, mu, logvar):
        NLL = self.criterion(recon_x, x)
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return NLL + KLD

    def forward(self, x):
        return self.model.encoder(x)

    def training_step(self, batch, batch_idx):
        recon_batch, mu, logvar = self.model(batch)
        loss = self.loss_function(recon_batch, batch, mu, logvar)
        return {
            'loss': loss,
            'log': {'train_loss': loss}
        }
    
    def validation_step(self, batch, batch_idx):
        recon_batch, mu, logvar = self.model(batch)
        loss = self.loss_function(recon_batch, batch, mu, logvar)
        return {
            'loss': loss
        }

    def validation_end(self, outputs):
        loss = sum([output['loss'] for output in outputs]) / len(outputs)
        return {
            'log': {
                'val_loss': loss
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


def run_fine_tuning(args, app_dims, app_train, app_test, sequences, encoders, name, onehot=False):
    app_data = {}
    app_data['application_train'] = app_train
    app_data['application_test'] = app_test
    loader_maker = LoaderMaker(app_data, sequences, args, onehot=onehot)

    # CV
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(app_train['SK_ID_CURR'], app_train['TARGET'])
    best_models = []
    for train_index, val_index in folds:
        train_dataloader = loader_maker.make(train_index)
        val_dataloader = loader_maker.make(val_index)
        model = LightningModel(
            PretrainedR2N(app_dims, args.n_hidden, args.n_main, encoders),
            nn.BCEWithLogitsLoss(),
            train_dataloader,
            val_dataloader,
            args
        )
        trainer = HomeCreditTrainer(name, args.n_epochs, args.patience)
        trainer.fit(model)
        best_model = load_model(model, name, trainer.logger.version)
        best_models.append(best_model)

    # Predict
    test_dataloader = loader_maker.make(index=None, train=False)
    df_submission = predict(best_models, test_dataloader)
    df_submission.to_csv(f'../submission/{name}.csv', index=False)

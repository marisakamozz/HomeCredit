import argparse
import pathlib
import math
import joblib
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from util import seed_everything, worker_init_fn, dump, read_sequences
from model import LSTMEncoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator
from plutil import load_model


class DIMLSTM(pl.LightningModule):
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


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, app, cat, cont):
        self.sk_id_curr = app[['SK_ID_CURR']]
        self.cat = cat
        self.cont = cont
        
    def __len__(self):
        return len(self.sk_id_curr)
        
    def __getitem__(self, idx):
        sk_id_curr = self.sk_id_curr.iloc[idx].values[0]
        return self.cat[sk_id_curr], self.cont[sk_id_curr]


def parse_args():
    parser = argparse.ArgumentParser(description='DIM(LSTM)')
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
            x = [data.to(device) for data in x]
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
    app_train = joblib.load('../data/03_powertransform/application_train.joblib')
    app_test = joblib.load('../data/03_powertransform/application_test.joblib')
    sequences = read_sequences('../data/04_sequence/')
    dims = joblib.load('../data/07_dims/dims03.joblib')
    dims.pop('application_train')
    dims.pop('application_test')

    for name, diminfo in dims.items():
        cat = sequences[f'{name}_cat']
        cont = sequences[f'{name}_cont']
        train_loader = torch.utils.data.DataLoader(
            SequenceDataset(app_train, cat, cont),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        test_loader = torch.utils.data.DataLoader(
            SequenceDataset(app_test, cat, cont),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        model = DIMLSTM(diminfo, args.n_hidden, train_loader, test_loader, args)
        logdir = '../logs/21_dimlstm'
        path = pathlib.Path(logdir) / name
        if not path.exists():
            path.mkdir(parents=True)
        logger = TensorBoardLogger(logdir, name=name)
        early_stopping = EarlyStopping(
            patience=args.patience,
            monitor='val_loss_main',
            mode='min'
        )
        filepath = pathlib.Path(logdir) / name / f'version_{logger.version}' / 'checkpoints'
        model_checkpoint = ModelCheckpoint(
            str(filepath),
            monitor='val_loss_main',
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
            SequenceDataset(app_train, cat, cont),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
        df_train = predict(name, best_model, train_loader_no_shuffle)
        df_test = predict(name, best_model, test_loader)
        df_encoding = pd.concat([df_train, df_test])
        dump(df_encoding, f'../data/21_dimlstm/{name}.joblib')


if __name__ == "__main__":
    main()

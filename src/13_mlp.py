"""
MLP

メインテーブルのみを利用してMLPで予測を行う。
事前学習＋ファインチューニングのベースライン。
"""

import argparse
import random
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from util import seed_everything, get_dims, HomeCreditDataset, OneHotDataset
from plutil import LightningModel, HomeCreditTrainer, load_model, predict
from model import MLP, MLPOneHot

def parse_args():
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--onehot', action='store_true')  # default=False
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=256)
    parser.add_argument('--n_epochs', action='store', type=int, default=30)
    parser.add_argument('--patience', action='store', type=int, default=5)
    parser.add_argument('--batch_size', action='store', type=int, default=1000)
    return parser.parse_args()

def worker_init_fn(worker_id):
    random.seed(worker_id)

def make_dataloader(application, index, batch_size, train=True, onehot=False):
    if onehot:
        loader = torch.utils.data.DataLoader(
            OneHotDataset(application, index=index),
            batch_size=batch_size,
            shuffle=train,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
    else:
        loader = torch.utils.data.DataLoader(
            HomeCreditDataset(application, index=index),
            batch_size=batch_size,
            shuffle=train,
            num_workers=6,
            worker_init_fn=worker_init_fn
        )
    return loader

def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.onehot:
        app_train = joblib.load('../data/05_onehot/application_train.joblib')
        app_test = joblib.load('../data/05_onehot/application_test.joblib')
        dims = get_dims({'application_train': app_train})
        _, _, cont_dim = dims['application_train']
        n_input = cont_dim
    else:
        app_train = joblib.load('../data/03_powertransform/application_train.joblib')
        app_test = joblib.load('../data/03_powertransform/application_test.joblib')
        dims = get_dims({'application_train': app_train})
        cat_dims, emb_dims, cont_dim = dims['application_train']
        n_input = emb_dims.sum() + cont_dim

    n_hidden = args.n_hidden

    # CV
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(app_train['SK_ID_CURR'], app_train['TARGET'])
    best_models = []
    for train_index, val_index in folds:
        train_dataloader = make_dataloader(app_train, train_index, args.batch_size, onehot=args.onehot)
        val_dataloader = make_dataloader(app_train, val_index, args.batch_size, onehot=args.onehot)
        if args.onehot:
            network = MLPOneHot(n_input, n_hidden)
        else:
            network = MLP(cat_dims, emb_dims, n_input, n_hidden)
        model = LightningModel(
            network,
            nn.BCEWithLogitsLoss(),
            train_dataloader,
            val_dataloader,
            args
        )
        name = '13_mlp-onehot' if args.onehot else '13_mlp-label'
        trainer = HomeCreditTrainer(name, args.n_epochs, args.patience)
        trainer.fit(model)

        best_model = load_model(model, name, trainer.logger.version)
        best_models.append(best_model)


    # Predict
    test_dataloader = make_dataloader(app_test, None, args.batch_size, train=False, onehot=args.onehot)
    df_submission = predict(best_models, test_dataloader)
    filename = '../submission/13_mlp-onehot.csv' if args.onehot else '../submission/13_mlp-label.csv'
    df_submission.to_csv(filename, index=False)


if __name__ == "__main__":
    main()

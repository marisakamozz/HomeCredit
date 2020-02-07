import argparse
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from util import seed_everything, read_all, get_dims, read_sequences, LoaderMaker
from plutil import LightningModel, HomeCreditTrainer, load_model, predict
from model import R2N

def parse_args():
    parser = argparse.ArgumentParser(description='R2N-LSTM')
    parser.add_argument('--onehot', action='store_true')  # default=False
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--lr', action='store', type=float, default=1e-3)
    parser.add_argument('--n_hidden', action='store', type=int, default=16)
    parser.add_argument('--n_main', action='store', type=int, default=32)
    parser.add_argument('--n_epochs', action='store', type=int, default=30)
    parser.add_argument('--patience', action='store', type=int, default=5)
    parser.add_argument('--batch_size', action='store', type=int, default=1000)
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.onehot:
        all_data = read_all(directory='../data/05_onehot')
        sequences = read_sequences(directory='../data/06_onehot_seq')
    else:
        all_data = read_all(directory='../data/03_powertransform')
        sequences = read_sequences(directory='../data/04_sequence')
    dims = get_dims(all_data)
    loader_maker = LoaderMaker(all_data, sequences, args, onehot=args.onehot)

    # CV
    name = 'R2N-LSTM-ONEHOT' if args.onehot else 'R2N-LSTM-LABEL'
    skf = StratifiedKFold(n_splits=5)
    folds = skf.split(all_data['application_train']['SK_ID_CURR'], all_data['application_train']['TARGET'])
    best_models = []
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
        trainer = HomeCreditTrainer(name, args.n_epochs, args.patience)
        trainer.fit(model)
        best_model = load_model(model, name, trainer.logger.version)
        best_models.append(best_model)

    # Predict
    test_dataloader = loader_maker.make(index=None, train=False)
    df_submission = predict(best_models, test_dataloader)
    filename = '../submission/14_r2n-lstm-onehot.csv' if args.onehot else '../submission/14_r2n-lstm-label.csv'
    df_submission.to_csv(filename, index=False)


if __name__ == "__main__":
    main()

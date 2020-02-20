"""
DIM (LSTM) で事前学習したモデルをファインチューニング
"""
import argparse
import joblib

from util import read_all
from plutil import load_model, DIMLSTMModule, run_fine_tuning

def parse_args():
    parser = argparse.ArgumentParser(description='DIM-LSTM FineTuning')
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
    app_train = joblib.load('../data/03_powertransform/application_train.joblib')
    app_test = joblib.load('../data/03_powertransform/application_test.joblib')
    sequences = read_all('../data/04_sequence')
    dims = joblib.load('../data/07_dims/dims03.joblib')
    app_dims = {}
    app_dims['application_train'] = dims.pop('application_train')
    app_dims['application_test'] = dims.pop('application_test')

    encoders = {}
    for name, diminfo in dims.items():
        model = DIMLSTMModule(diminfo, args.n_hidden, None, None, args)
        model = load_model(model, name, logdir='../logs/21_dimlstm')
        encoder = model.encoder
        encoders[name] = encoder
    
    run_fine_tuning(args, app_dims, app_train, app_test, sequences, encoders, '41_dimlstm')


if __name__ == "__main__":
    main()

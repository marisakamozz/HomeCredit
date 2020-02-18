"""
VAE (LSTM) で生成した特徴量でLightGBMで予測
"""
import argparse
import joblib

from util import read_all, run_lgb

def parse_args():
    parser = argparse.ArgumentParser(description='VAE-LSTM FEATURE')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--objective', action='store', type=str, default='binary')
    parser.add_argument('--n_estimators', action='store', type=int, default=10000)
    parser.add_argument('--early_stopping', action='store', type=int, default=100)
    parser.add_argument('--device', action='store', type=str, default='gpu')
    return parser.parse_args()

def main():
    args = parse_args()
    app_train = joblib.load('../data/01_labelencoding/application_train.joblib')
    app_test = joblib.load('../data/01_labelencoding/application_test.joblib')
    features = read_all('../data/22_vaelstm')
    for feature in features.values():
        app_train = app_train.merge(feature)
        app_test = app_test.merge(feature)
    
    run_lgb(args, app_train, app_test, '33_vaelstm')


if __name__ == "__main__":
    main()

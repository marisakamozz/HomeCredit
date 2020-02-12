"""
集計特徴量

featuretoolsを利用して作成した集計特徴量を用いてLightGBMで予測を行う。
"""

import argparse
import joblib
import lightgbm
import mlflow

from util import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description='featuretools')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--objective', action='store', type=str, default='binary')
    parser.add_argument('--device', action='store', type=str, default='gpu')
    return parser.parse_args()

def main(args):
    seed_everything(args.seed)

    feature_matrix = joblib.load('../data/02_featuretools/feature_matrix.joblib')
    df_train = feature_matrix.dropna(subset=['TARGET'])
    df_test = feature_matrix[feature_matrix['TARGET'].isnull()].drop('TARGET', axis=1)

    x_train = df_train.drop('TARGET', axis=1)
    y_train = df_train['TARGET']
    train_set = lightgbm.Dataset(x_train, y_train)

    params = vars(args)
    mlflow.log_params(params)

    eval_hist = lightgbm.cv(params, train_set, metrics='auc')
    for i, metric in enumerate(eval_hist['auc-mean']):
        mlflow.log_metric('auc', metric, step=i)

    model = lightgbm.train(params, train_set)
    x_test = df_test
    y_pred = model.predict(x_test)
    df_submission = x_test[['SK_ID_CURR']].copy()
    df_submission['TARGET'] = y_pred
    df_submission.to_csv('../submission/12_featuretools.csv', index=False)


if __name__ == "__main__":
    mlflow.set_tracking_uri('../logs/mlruns')
    mlflow.set_experiment('HomeCredit')
    with mlflow.start_run(run_name='featuretools') as run:
        main(parse_args())

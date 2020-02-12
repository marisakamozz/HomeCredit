"""
ベースラインモデル

メインテーブルのみを利用してLightGBMで予測を行う。
"""
import argparse
import joblib
import lightgbm as lgb
import mlflow

from util import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--seed', action='store', type=int, default=1234)
    parser.add_argument('--objective', action='store', type=str, default='binary')
    parser.add_argument('--device', action='store', type=str, default='gpu')
    return parser.parse_args()

def main(args):
    seed_everything(args.seed)
    # Train
    df_train = joblib.load('../data/01_labelencoding/application_train.joblib')
    x_train = df_train.drop('TARGET', axis=1)
    y_train = df_train['TARGET']
    train_set = lgb.Dataset(x_train, y_train)

    params = vars(args)
    mlflow.log_params(params)

    eval_hist = lgb.cv(params, train_set, metrics='auc')
    for i, metric in enumerate(eval_hist['auc-mean']):
        mlflow.log_metric('auc', metric, step=i)

    # Predict
    df_test = joblib.load('../data/01_labelencoding/application_test.joblib')
    x_test = df_test
    model = lgb.train(params, train_set)
    y_pred = model.predict(x_test)
    df_submission = x_test[['SK_ID_CURR']].copy()
    df_submission['TARGET'] = y_pred

    path = '../submission/11_baseline.csv'
    df_submission.to_csv(path, index=False)
    mlflow.log_artifact(path)

if __name__ == "__main__":
    mlflow.set_tracking_uri('../logs/mlruns')
    mlflow.set_experiment('HomeCredit')
    with mlflow.start_run(run_name='baseline') as run:
        main(parse_args())

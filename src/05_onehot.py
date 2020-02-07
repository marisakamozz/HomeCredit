"""
前処理その５
・カテゴリー列をOne-Hot Encodingする

これらの処理を行ったものがニューラルネットワーク（エンベディング層なし）の入力となる
"""
from multiprocessing import Pool
import pandas as pd
from util import read_all, dump

def process(item):
    name, df = item
    df = pd.get_dummies(df)
    dump(df, f'../data/05_onehot/{name}.joblib')

def main():
    all_data = read_all(directory='../data/03_powertransform')
    app_train = all_data.pop('application_train')
    app_test = all_data.pop('application_test')
    data = app_train.append(app_test, sort=False)
    data = pd.get_dummies(data)
    app_train = data.dropna(subset=['TARGET'])
    app_test = data[data['TARGET'].isnull()].drop('TARGET', axis=1)
    dump(app_train, '../data/05_onehot/application_train.joblib')
    dump(app_test, '../data/05_onehot/application_test.joblib')
    with Pool(6) as pool:
        pool.map(process, list(all_data.items()))

if __name__ == "__main__":
    main()

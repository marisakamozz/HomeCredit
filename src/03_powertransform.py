"""
前処理その3
・数値列を PowerTransformer でスケール変換する
・数値列の欠損値を判定するカテゴリー列を追加する
・すべて同じ情報を含む列を削除する
・数値列の欠損値を 0（＝平均値）で埋める

これらの処理を行ったものがニューラルネットワーク（エンベディング層あり）の入力となる
"""

from multiprocessing import Pool
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from util import read_all, save_df_with_dtypes

def power_transform(df):
    for column in df.select_dtypes(include=['int', 'float']).columns:
        if column == 'TARGET' or column.startswith('SK_ID'):
            continue
        if column == 'AMT_INCOME_TOTAL':
            encoder = PowerTransformer(method='box-cox')
        else:
            encoder = PowerTransformer(method='yeo-johnson')
        df[column] = encoder.fit_transform(df[[column]])
        df[column] = df[column].astype('float32')
    return df

def fillna(df: pd.DataFrame):
    for column in df.columns[df.isnull().any()]:
        if column == 'TARGET':
            continue
        df[column + '_NAN'] = df[column].isnull().astype(int).astype('category')
        df[column].fillna(0, inplace=True)
    return df

def drop_same_columns(df):
    drop_cols = set()
    for c1 in df.columns:
        if c1 in drop_cols:
            continue
        for c2 in df.select_dtypes(df[c1].dtype).columns:
            if c1 == c2 or c2 in drop_cols:
                continue
            try:
                if (df[c1] == df[c2]).all():
                    drop_cols.add(c2)
            except TypeError:
                continue
    df.drop(drop_cols, axis=1, inplace=True)
    return df

def process(item):
    name, df = item
    df = power_transform(df)
    df = fillna(df)
    df = drop_same_columns(df)
    save_df_with_dtypes(df, f'../data/03_powertransform/{name}.csv')


if __name__ == "__main__":
    datas = read_all()
    app_train = datas.pop('application_train')
    app_test = datas.pop('application_test')
    app = app_train.append(app_test, sort=False)
    app = power_transform(app)
    app = fillna(app)
    app = drop_same_columns(app)
    app_train = app.dropna(subset=['TARGET'])
    app_test = app[app['TARGET'].isnull()].drop('TARGET', axis=1)
    save_df_with_dtypes(app_train, '../data/03_powertransform/application_train.csv')
    save_df_with_dtypes(app_test, '../data/03_powertransform/application_test.csv')

    with Pool(6) as pool:
        pool.map(process, list(datas.items()))

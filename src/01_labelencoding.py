"""
前処理１
・「DAYS_...」に含まれる外れ値 365243 をnp.nanに置き換える
・0/1のみの数値列をカテゴリーとして扱う
・カテゴリー列をLabelEncodingする（欠損値もカテゴリーの一つとして扱う）

これらの処理を行ったものがLightGBMの入力となる
"""

import pathlib
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util import dump

def preprocess(df):
    # outlier
    for col in df.columns:
        if "DAYS" in col:
            df[col] = df[col].replace({365243: np.nan})
    
    # treat boolean as category
    for column in df.select_dtypes('int').columns:
        if df[column].min() == 0 and df[column].max() == 1:
            df[column] = df[column].astype('category')

    # label encoding
    for column in df.select_dtypes('object').columns:
        df[column].fillna('ZZZZ', inplace=True)
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        df[column] = df[column].astype('category')
    return df

def save(df, filename):
    dump(df, f'../data/01_labelencoding/{filename}.joblib')

def process(file):
    print(file.name)
    df = pd.read_csv(file)
    df = preprocess(df)
    save(df, file.stem)

def main():
    app_train = pd.read_csv('../input/application_train.csv')
    app_test = pd.read_csv('../input/application_test.csv')
    app = app_train.append(app_test, sort=False)
    app = preprocess(app)
    app_train = app.dropna(subset=['TARGET'])
    app_test = app[app['TARGET'].isnull()].drop('TARGET', axis=1)
    save(app_train, 'application_train')
    save(app_test, 'application_test')

    except_files = [
        'HomeCredit_columns_description.csv',
        'sample_submission.csv',
        'application_train.csv',
        'application_test.csv'
    ]
    target_files = [
        file for file in pathlib.Path('../input').glob('*.csv') if file.name not in except_files
    ]
    
    with Pool(6) as pool:
        pool.map(process, target_files)


if __name__ == "__main__":
    main()

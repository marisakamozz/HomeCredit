import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

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

datas = read_all()
app_train = datas.pop('application_train')
app_test = datas.pop('application_test')
data = app_train.append(app_test, sort=False)
data = power_transform(data)
app_train = data.dropna(subset=['TARGET'])
app_test = data[data['TARGET'].isnull()].drop('TARGET', axis=1)
save_df_with_dtypes(app_train, '../data/04_powertransform/application_train.csv')
save_df_with_dtypes(app_test, '../data/04_powertransform/application_test.csv')

for name, df in datas.items():
    df = power_transform(df)
    save_df_with_dtypes(df, f'../data/04_powertransform/{name}.csv')
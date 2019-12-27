import pathlib
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    df.to_csv(f'../data/02_labelencoding/{filename}.csv', index=False)
    with open(f'../data/02_labelencoding/dtypes_{filename}.pkl', mode='wb') as file:
        pickle.dump(df.dtypes.to_dict(), file)

df_train = pd.read_csv('../input/application_train.csv')
df_test = pd.read_csv('../input/application_test.csv')
data = df_train.append(df_test, sort=False)
data = preprocess(data)
df_train = data.dropna(subset=['TARGET'])
df_test = data[data['TARGET'].isnull()].drop('TARGET', axis=1)
save(df_train, 'application_train')
save(df_test, 'application_test')

except_files = [
    'HomeCredit_columns_description.csv',
    'sample_submission.csv',
    'application_train.csv',
    'application_test.csv'
]

for file in pathlib.Path('../input').glob('*.csv'):
    if file.name in except_files:
        continue
    else:
        print(file.name)
        df = pd.read_csv(file)
        df = preprocess(df)
        save(df, file.stem)

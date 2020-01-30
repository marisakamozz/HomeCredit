"""
前処理その４
・One-Hot Encoding した履歴テーブルの情報を SK_ID_CURR をキー、50件の np.ndarray を値として保有するdictionaryに変換する
・50件に満たない場合はすべて0の行を追加して50件とする

ニューラルネットワーク（エンベディング層なし）の処理高速化のため
"""
import pickle
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

from util import read_all, sort_keys, expand
from model import max_len


def process(df_sk_id_curr, item):
    name, df = item
    print(f'--- {name} ---')
    cont = {}
    drop_cols = [column for column in df.columns if column.startswith('SK_ID')]
    for sk_id_curr in tqdm(df_sk_id_curr['SK_ID_CURR']):
        data = df.query('SK_ID_CURR == @sk_id_curr').sort_values(sort_keys[name]).tail(max_len)
        data = data.drop(drop_cols, axis=1)
        cont[sk_id_curr] = expand(data.values, max_len)
    with open(f'../data/06_onehot_seq/{name}.pkl', mode='wb') as file:
        pickle.dump(cont, file)

def main():
    all_data = read_all('../data/05_onehot')

    app_train = all_data.pop('application_train')
    app_test = all_data.pop('application_test')
    df_sk_id_curr = pd.concat([app_train[['SK_ID_CURR']], app_test[['SK_ID_CURR']]])
    # df_sk_id_curr = df_sk_id_curr.head(100)

    bureau = all_data['bureau']
    bureau_balance = all_data['bureau_balance']
    all_data['bureau_balance'] = pd.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], bureau_balance, on='SK_ID_BUREAU')

    id_list = [df_sk_id_curr] * len(all_data)
    with Pool(6) as pool:
        pool.starmap(process, zip(id_list, list(all_data.items())))


if __name__ == "__main__":
    main()

# coding : utf-8

import pandas as pd
import sqlite3
import pandas.io.sql as psql
import pickle

def df_to_csv(df, file_name, index=True, header=True, sep=',', encoding='shift_jis'):
    with open(file_name, 'w', encoding=encoding) as f:
        df.to_csv(f, index=index, header=header, sep=sep)


def df_to_sqlite(df, db_name, table_name, index=True):
    with sqlite3.connect(db_name) as conn:
        psql.to_sql(df, table_name, conn, index=index, if_exists='replace')


def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def csv_to_df(file_name, index_col=None, header='infer', sep=',', encoding='shift_jis'):
    with open(file_name, 'r', encoding=encoding) as f:
        df = pd.read_csv(f, index_col=index_col, header=header, sep=sep)
    return df


def sqlite_to_df(file_name, table_name, index_col=None):
    with sqlite3.connect(file_name) as conn:
        sql = 'select * from ' + table_name
        df = psql.read_sql(sql, conn, index_col=index_col)
    return df


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj
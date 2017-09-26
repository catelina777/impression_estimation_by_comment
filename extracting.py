'''コメントの特徴量を抽出するプログラム'''
# coding: utf-8

import pandas as pd
from parces.comment_parce import CommentParce
import glob
import os

# file_path = './comment_source/'
# file_path = './filtered_comment_source_audio/'
# file_path = './filtered_comment_source_other/'

output = 'feature_values.csv'


def extracting(word_class=["形容詞"], max_features=30, vector='tfidf', comment_directory_path='./comment_source/', output_feature_name='feature_values.csv'):

    files = glob.glob(comment_directory_path + '*.txt')
    texts = list(map(read_text, files))
    index = list(map(get_name, files))

    cp = CommentParce(word_class=word_class,
                      max_features=max_features, vector=vector)

    values = cp.fit_parce(texts)
    df = pd.DataFrame(values, index=index, columns=cp.feature_names)
    df.index.name = '動画ID'

    with open(output_feature_name, 'w', encoding='utf_8_sig') as f:
        df.to_csv(f)


def main():
    pass


def read_text(file):
    with open(file, 'r') as f:
        text = f.read()
    return text


def get_name(file):
    name, ext = os.path.splitext(os.path.basename(file))
    return name


if __name__ == '__main__':
    main()

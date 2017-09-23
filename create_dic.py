# -*- coding: utf-8 -*-

from gensim import models
import csv

# music_dictionary = [
#         "調教", "歌", "聴", "音", "曲", "歌詞", "メロディ",
#         "BGM", "耳", "歌って", "リズム", "音圧", "ドラム",
#         "イントロ", "聞", "声"
#     ]

seed_dictionary = [
    "調教", "歌", "聴", "音", "曲", "歌詞",
    "メロディ", "BGM", "耳", "歌って", "リズム",
    "音圧", "ドラム", "イントロ", "聞", "声",

    "コメ", "周年", "こめ", "おめでとう", "巡回", "？", "いけ",
    "初見", "評価", "再生", "マイリス",
    "支援", "うぽつ", "職人", "今日", "誕生日", "弾幕", "久しぶり",
    "gj"
    ]

# load model data
model = models.KeyedVectors.load_word2vec_format('./nico_vec.bin', binary=True, unicode_errors='ignore')

dic = []

for seed_value in seed_dictionary:
    try:
        related_values = model.most_similar(positive=[seed_value], topn=20)

    except KeyError:
        print("not in vocabulary = " + seed_value)
        continue

    for value in related_values:
        dic.append(value[0])

dic = dic + seed_dictionary

f = open('dic.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(dic)
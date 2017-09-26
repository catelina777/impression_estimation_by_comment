# coding : utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import MeCab    #形態素解析に用いるモジュール
mecab = MeCab.Tagger('-Ochasen -d /usr/workspace/mecab-ipadic-neologd') #-d以下は参照URL先の辞書を使用している
from tqdm import tqdm
import pickle

import glob
import os

class CommentParce:

    def __init__(self, max_df=1.0, word_class=['形容詞'], max_features=None, vector='tfidf'):
        self.max_df = max_df
        self.max_features = max_features
        self.vector = vector
        self.word_class = word_class

        self.vectorizer = None
        self.feature_names = None
        self._splited = None

    def _split_text(self, text):
        node = mecab.parseToNode(text)
        words = []
        while node:
            try:
                info = node.feature.split(',')
                word = info[6]
                pos = info[0]
                if pos in self.word_class:
                    words.append(word)
                
                node = node.next

            except AttributeError:
                break
        
        return ' '.join(words)


    def _split_texts(self, texts, viewing=True):
        splited = []
        if viewing : texts = tqdm(texts)
        
        for text in texts:
            splited.append(self._split_text(text))

        self._splited = splited


    def fit_parce(self, texts, splitting=True):
        if splitting:
            self._split_texts(texts)
        self.fit(texts=None, splitting=False)
        bag = self.parce(texts=None, splitting=False)

        return bag
    

    def fit(self, texts, splitting=True):
        if splitting:
            self._split_texts(texts)
        if self.vector == 'tfidf':
            vectorizer = TfidfVectorizer(
                use_idf = True,
                norm = 'l2',
                max_df = self.max_df,
                max_features = self.max_features,
                smooth_idf = True
            )
        elif self.vector == 'count':
            vectorizer = CountVectorizer(
                max_df = self.max_df,
                max_features = self.max_features
            )

        vectorizer.fit(self._splited)
        
        self.vectorizer = vectorizer
        self.feature_names = vectorizer.get_feature_names()
        

    def parce(self, texts, splitting=True):
        if splitting:
            self._split_texts(texts, viewing=False)
        bag = self.vectorizer.transform(self._splited)
        bag = bag.toarray()

        return bag




def main():
    file_path = '/Users/abe/Projects/impression/sources/comment_source/'
    files = glob.glob(file_path + '*.txt')
    texts, index = get_text_matrix(files)
    cp = CommentParce(max_df=0.5, max_features=40)
    values = cp.fit_parce(texts)
    print(values[0])
    print(len(values[0]))

    #print(values)
    print(cp.feature_names)
    #print(cp.parce([texts[0]]))
    #print(index[0])
    

def get_text_matrix(files):
    texts = []
    index = []
    for file in files:
        with open(file, 'r') as f:
            texts.append(f.read())
        name, ext = os.path.splitext(os.path.basename(file))
        index.append(name)
    
    return texts, index

if __name__ == '__main__':
    main()


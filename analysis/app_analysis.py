#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:37:21 2019

@author: marcogancitano
"""

from datetime import date
import os

import numpy as np
import pandas as pd

files = np.array(os.listdir('../data/raw'))
files = files[[".csv" in file for file in files]]

news_data = pd.DataFrame()
for file in files:
    news_source = file.split('.')[0]
    df = pd.read_csv('../data/raw/' + file)
    df['source'] = news_source
    df = df.loc[~(df.loc[:,["article",'title']].isna().sum(axis = 1) > 0)]
    news_data = pd.concat([news_data,df])

"""
file = 'cnn.csv'
df = pd.read_csv('../data/raw/' + file)

df = df.loc[~(df.loc[:,["url","article",'title']].isna().sum(axis = 1) > 0)]
df = df.reset_index()


published_dates = [None] * len(df)
for i,row in df.iterrows():
    split_url = row.url.split('.com')[1].split('/')
    published_dates[i] = (str(date(int(split_url[1]),int(split_url[2]),int(split_url[3]))))
    
df.loc['published_date'] = published_dates
df.to_csv('../data/raw/' + file,index=False)
"""

from collections import defaultdict
import itertools

from gensim.corpora.dictionary import Dictionary

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np

column_to_process = 'article'

articles_to_search = list(news_data[column_to_process])

stop_words = stopwords.words('English')
wordnet_lemmatizer = WordNetLemmatizer()
articles = [None] * len(articles_to_search)
i = 0
for text in articles_to_search:
    articles[i] = [wordnet_lemmatizer.lemmatize(w) for w in word_tokenize(text.lower()) if (w.isalpha() and w not in stop_words)]
    i += 1
    
dictionary = Dictionary(articles)
corpus = np.array([dictionary.doc2bow(article) for article in articles])

from textblob import TextBlob
from tqdm import tqdm

ents = []
sentiments = []
for i,row in tqdm(breitbart.iterrows()):
    txt = TextBlob(row.article_text)
    sentiments.append([sent.sentiment for sent in txt.sentences])
        
    #breitbart.loc[i,'polarity'] = sentiment.polarity
    #breitbart.loc[i,'subjectivity'] = sentiment.subjectivity
    
    
    dd = {}
    for ent,ent_label in pos_tag(txt.words):
        ent = ent.singularize()
        if ent_label == "NNP":
            if ent in dd:
                dd[ent] += 1
            else:
                dd[ent] = 1
    ents.append(dd)
    
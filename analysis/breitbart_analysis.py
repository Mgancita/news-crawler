#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:37:21 2019

@author: marcogancitano
"""

# -*- coding: utf-8 -*-
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
    df = df.loc[~(df.loc[:,["article",'author','title']].isna().sum(axis = 1) > 0)]
    news_data = pd.concat([news_data,df])

news_data = news_data.reset_index()


#breitbart = pd.read_csv('../data/titles.csv')
#breitbart = breitbart.loc[~(breitbart.loc[:,["article",'author','title']].isna().sum(axis = 1) > 0)]

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

ents = [None] * len(articles_to_search)
sentiments = [None] * len(articles_to_search)
for i,row in tqdm(news_data.iterrows()):
    txt = TextBlob(row.article)
    sentiment = [sent.sentiment for sent in txt.sentences]
    temp_store = [(sentence[0],sentence[1]) for sentence in sentiment]
    sentiments[i] = temp_store
        
    
    #breitbart.loc[i,'polarity'] = sentiment.polarity
    #breitbart.loc[i,'subjectivity'] = sentiment.subjectivity
    
    
    """
    dd = {}
    for ent,ent_label in pos_tag(txt.words):
        ent = ent.singularize()
        if ent_label == "NNP":
            if ent in dd:
                dd[ent] += 1
            else:
                dd[ent] = 1
    ents[i] = dd
    """

sent_values = []
for sentiment in sentiments:
    temp_store = []
    for sentence in sentiment:
        temp_store.append((sentence[0],sentence[1]))
    sent_values.append(temp_store)
    
for i,row in (news_data.iterrows()):
    if row['source'] == 'cnn':
        cut = row['url'].split('com/')[1].split('/')
        news_data.at[i,'published_date'] = str(datetime(int(cut[0]),int(cut[1]),int(cut[2])).date())    

news_data.to_csv('../data/refined/news_data.csv',index=False)    

import pickle
pickle.dump(dictionary, open( "../data/refined/dictionary.p", "wb" ))
pickle.dump(corpus, open( "../data/refined/corpus.p", "wb" ))
pickle.dump(ents, open( "../data/refined/entities.p", "wb" ))
pickle.dump(sentiments, open( "../data/refined/sentiments.p", "wb" ))

from datetime import datetime
import numpy as np
import pandas as pd

from textblob import TextBlob
from tqdm import tqdm

import time

t = time.time()
import pickle
news_data = pd.read_csv("../data/refined/news_data.csv")
#dictionary = pickle.load(open( "../data/refined/dictionary.p", "rb" ))
#corpus = pickle.load(open( "../data/refined/corpus.p", "rb" ))
#ents = pickle.load(open( "../data/refined/entities.p", "rb" ))
sentiments = pickle.load(open("../data/refined/sentiments.p","rb"))
print(time.time()-t)

##TODO
#Make an algorithm which looks at sentence by sentence for sentiment instead of entire article

search_term = "Trump"
idx = list(np.where(news_data.article.str.contains(search_term)))[0]
news_data_idxed = news_data.loc[idx]
search_sentiment = [sentiments[i] for i in idx]


pols = [None] * len(search_sentiment)
subs = [None] * len(search_sentiment)
for i in tqdm(range(len(search_sentiment))):
    sentiment = search_sentiment[i]
    sents = TextBlob(news_data_idxed.article.iloc[i]).sentences
    sent_idx = list(np.where([search_term in sent for sent in sents])[0])
    og_idx = sent_idx[:]
    for idx in og_idx:
        lb = idx-1
        ub = idx+1
        if lb > 0 and lb not in sent_idx:
            sent_idx.append(lb)
        if ub < len(sentiment) and ub not in sent_idx:
            sent_idx.append(ub)
    
    pols[i] = np.mean([sentiment[j][0] for j in sent_idx])
    subs[i] = np.mean([sentiment[j][1] for j in sent_idx])    
    

sentiment_by_time = pd.DataFrame(list(zip(news_data_idxed.published_date,news_data_idxed.source,pols,subs)),columns = ['date','source','polarity','subjectivity'])
sentiment_by_time = sentiment_by_time.dropna()

for i,row in sentiment_by_time.iterrows():
    sentiment_by_time.at[i,'date'] = str(datetime.strptime(row['date'],"%Y-%m-%d").replace(day=1).date())


#print((pols * (news_data_idxed.shares/news_data_idxed.shares.sum())).sum())
#print((subs * (news_data_idxed.shares/news_data_idxed.shares.sum())).sum())

#sentiment_by_time['month'] = [datetime.strptime(date_to_clean, "%Y-%m-%d").month for date_to_clean in sentiment_by_time.date]
#sentiment_by_time['year'] = [datetime.strptime(date_to_clean, "%Y-%m-%d").year for date_to_clean in sentiment_by_time.date]


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for key, grp in sentiment_by_time.groupby(['source']):

    print(key)
    plot_data=grp.groupby('date').subjectivity.mean()
    ax = plot_data.plot(ax=ax, kind='line', label=key)

plt.legend()
plt.show()


import matplotlib.pyplot as plt
plt.plot()
plt.show()


print(((breitbart_idxed['polarity'] * breitbart_idxed['shares'])/breitbart_idxed['shares'].sum()).sum())
print(((breitbart_idxed['subjectivity'] * breitbart_idxed['shares'])/breitbart_idxed['shares'].sum()).sum())




search_corpus = list(corpus[idx])

total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(search_corpus):
    total_word_count[word_id] += word_count
    
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 

print("Popular words:")
for word_id, word_count in sorted_word_count[:10]:
    print(dictionary.get(word_id), word_count)
    
cut_ents = [ents[i] for i in idx]
merged_dict = {}
for ent_dict in cut_ents:
    for ent in ent_dict:
        if ent in merged_dict:
            merged_dict[ent] += 1
        else:
            merged_dict[ent] = 1

print()
print("Popular Entities:")
sorted_x = sorted(merged_dict.items(), key=lambda kv: kv[1],reverse = True)
for x in sorted_x[:10]:
    print("%s: %i" %(x[0],x[1]))
    
"""
from gensim import models
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=5) 
lda.print_topics(5)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5) 
lsi.print_topics(5)
"""
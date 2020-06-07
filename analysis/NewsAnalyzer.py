#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 12:22:28 2019

@author: marcogancitano
"""
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from textblob import TextBlob

def find_data(search_term, search_column):
    idx = list(np.where(news_data[search_column].str.contains(search_term)))[0]
    news_data_idxed = news_data.loc[idx]
    search_sentiment = [sentiments[i] for i in idx]
    return news_data_idxed, search_sentiment

def get_term_sentiment(news_articles, sentiment_data, search_term):
    pols = [None] * len(sentiment_data)
    subs = [None] * len(sentiment_data)
    for i in range(len(sentiment_data)):
        sentiment = sentiment_data[i]
        sents = TextBlob(news_articles.iloc[i]).sentences
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
    return pols, subs
    
class NewsAnalyzer:
    
    def __init__(self, news_data, sentiment_data):
        self.news_data = news_data
        self.sentiment_data = sentiment_data
        
    def __call__(self, search_term, search_column, plot = False):
        news_data_idxed, sentiment_data_idxed = find_data(search_term,
                                                         search_column)
        polarity, subjectivity = get_term_sentiment(
                news_data_idxed[search_column],
                sentiment_data_idxed,
                search_term)
        
        if plot:
            sentiment_by_time = pd.DataFrame(
                    list(
                            zip(
                                    news_data_idxed.published_date,
                                    news_data_idxed.source,polarity,
                                    subjectivity
                                    )
                            ),
                            columns = [
                                    'date','source','polarity','subjectivity'
                                    ]
                            )
            sentiment_by_time = sentiment_by_time.dropna()
            for i,row in sentiment_by_time.iterrows():
                sentiment_by_time.at[i,'date'] = str(
                        datetime.strptime(
                                row['date'],"%Y-%m-%d").replace(day=1).date()
                        )
            
            fig, ax = plt.subplots()
            for calc in ['subjectivity','polarity']:
                for key, grp in sentiment_by_time.groupby(['source']):
                    plot_data=grp.groupby('date')[calc].mean()
                    ax = plot_data.plot(ax=ax, kind='line', label=key)
                
                plt.legend()
                plt.show()
            
        return polarity, subjectivity
        
    
if __name__ == "__main__":
    news_data = pd.read_csv("../data/refined/news_data.csv")
    sentiments = pickle.load(open("../data/refined/sentiments.p","rb"))
    
    analyzer = NewsAnalyzer(news_data,sentiments)
    p,s = analyzer('Clinton','article',plot=True)

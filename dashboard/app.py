# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools

import dash
import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd
import pickle

breitbart = pd.read_csv("../data/titles.csv")
dictionary = pickle.load(open( "../data/dictionary.p", "rb" ))
corpus = pickle.load(open( "../data/corpus.p", "rb" ))
ents = pickle.load(open( "../data/entities.p", "rb" ))

search_term = "Muslim"
idx = list(np.where(breitbart.article_text.str.contains(search_term)))[0]

search_corpus = list(corpus[idx])

total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(search_corpus):
    total_word_count[word_id] += word_count
    
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 
    
word_x = [dictionary[tup[0]] for tup in sorted_word_count[:10]]
word_y = [tup[1] for tup in sorted_word_count[:10]]

cut_ents = [ents[i] for i in idx]
merged_dict = {}
for ent_dict in cut_ents:
    for ent in ent_dict:
        if ent in merged_dict:
            merged_dict[ent] += 1
        else:
            merged_dict[ent] = 1

sorted_x = sorted(merged_dict.items(), key=lambda kv: kv[1],reverse = True)
ent_x = [tup[0] for tup in sorted_x[:10]]
ent_y = [tup[1] for tup in sorted_x[:10]]


app = dash.Dash()
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='News Sentiment Analysis',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='A web application for seeing sentiment around terms by news sources', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(
        id='Graph1',
        figure={
            'data': [
                {'x': ent_x, 'y': ent_y, 'type': 'bar', 'name': 'Entities'}
                #{'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
])
    
if __name__ == '__main__':
    app.run_server(debug=True)
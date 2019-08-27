#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:38:51 2019

@author: tauro
"""

import json

import pandas as pd


df = pd.read_json('ChatbotCorpus.json')


class DataGetter():
    
    def __init__(self, df):
        
        self.df = df
        
        self.df['text'] = None
        self.df['intent'] = None
        self.df['entities'] = None
        self.df['training'] = None
        
        
        for index, row in self.df.iterrows():
            for key, value in self.df.iloc[index].sentences.items():
                self.df.iloc[index][key] = value
                
        
        #self.sentences = self.df.sentences.split()
        
        
        
        
        
DataGetter(df)



ent_types = list(set([dict_['entity'] for sentence in df.entities for dict_ in sentence]))


sentences_tokenised = df.text.apply(lambda x: x.split(' '))

ent_start_end = df.entities.apply(lambda x: [(w['start'], w['stop']) for w in x])

ent_sentences = sentences_tokenised.apply(lambda x: ['O' for word in x])

for i, (sent, start_end) in enumerate(zip(sentences_tokenised, ent_start_end)):
    
    for j, word in enumerate(sent):
        if j in start_end:
            ent_sentences[i][j] = 'ENT'
        




# Intent classification
        
class IntentClassifier:
    
    def __init__(self):
        self.df = df



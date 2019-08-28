#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:38:51 2019

@author: tauro
"""

import json

import pandas as pd

import re



#df = pd.read_json('ChatbotCorpus.json')


class DataGetter():
    
    def __init__(self, filename):
        
        self.df = pd.read_json(filename)
        
        self.df['text'] = None
        self.df['intent'] = None
        self.df['entities'] = None
        self.df['training'] = None
        
        
        for index, row in self.df.iterrows():
            for key, value in self.df.iloc[index].sentences.items():
                self.df.iloc[index][key] = value
            
    
    
    
    
    def get_sentences(self):
        self.sentences = self.df.text.tolist()
        return self.sentences





    def clean_and_tokenise(self):   
        
        self.get_sentences()
    
        for i, sent in enumerate(self.sentences):
            self.sentences[i] = re.sub('[?.!]+', "", sent)            
        
        self.sentences_tokenised = list(map(lambda x: re.split(" |(-)|'|(:)|(,)", x), self.sentences))
        
        for i, sent in enumerate(self.sentences_tokenised):
            self.sentences_tokenised[i] = list(filter(None, sent))
            
        return self.sentences_tokenised
            


    
    def get_unique_entities(self):
        self.unique_ents = list(set([dict_['entity'] for sentence in self.df.entities for dict_ in sentence]))
        return self.unique_ents

        
        
        
    
    def get_entities(self):
        
        self.clean_and_tokenise()
        
        self.entitites_start_stop = df.entities.apply(lambda x: [(w['start'], w['stop']) for w in x])
    
        self.entity_sequence = list(map(lambda x: ['O' for word in x], self.sentences_tokenised))
        

        for i, sent in enumerate(self.sentences_tokenised):
            for ent in self.df.iloc[i].entities:
                start = ent['start']
                stop = ent['stop']
                
                entity = ent['entity']
                                
                try:
                    self.entity_sequence[i][start:stop+1] = [entity for i in range(len(self.entity_sequence[i][start:stop+1]))]
                except IndexError:
                    start -= 1
                    stop -= 1
                    
                    self.entity_sequence[i][start:stop+1] = [entity for i in len(self.entity_sequence[i][start:stop+1])]


        return self.entity_sequence
        
        
        
data = DataGetter('ChatbotCorpus.json')

df = data.df

unique_entities = data.get_unique_entities()

sentences = data.clean_and_tokenise()




 
    
    






#from nltk.tokenize import RegexpTokenizer
#
#tkr = RegexpTokenizer()






#def text_cleaner(sents):
#    
#    pattern_to_exclude = ['[^!.?]+']
#    
#    pattern = re.compile('[?.!]+')
#    
#    for i, sent in enumerate(sents):
#        print([token for token in sent if pattern.search(token)])
#        #sents[i] = [token for token in sent if not pattern.search(token)]
#        
#    return sents
#
#
#sentences = text_cleaner(sentences_tokenised)
    
    #sents = list(map(lambda x))
    
    
    
    
    
   
    
    

#sentences_tokenised = [token.replace(" ", "") for sent in sentences_tokenised for token in sent]


#error_annotations = [33, 89, 106]
#
#for index in error_annotations:
#    for ent in df.iloc[index].entities:
#        ent['start'] -= 1
#        ent['stop'] -= 1


#for i, (sent, start_end) in enumerate(zip(sentences_tokenised, ent_start_end)):
#    
#    for j, word in enumerate(sent):
#        if j in start_end:
#            ent_sentences[i][j] = 'ENT'
        












# Intent classification
        
class IntentClassifier:
    
    def __init__(self):
        self.df = df



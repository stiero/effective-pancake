#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:38:51 2019

@author: tauro
"""

import json

import pandas as pd

import re

import matplotlib.pyplot as plt
import seaborn as sns

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
            self.sentences[i] = re.sub('[.!]+', "", sent)            
        
        self.sentences_tokenised = list(map(lambda x: re.split(" |(-)|'|(:)|(,)|(\?)", x), self.sentences))
        
        for i, sent in enumerate(self.sentences_tokenised):
            self.sentences_tokenised[i] = list(filter(None, sent))
            
        return self.sentences_tokenised
            


    
    def get_unique_entities(self):
        self.unique_ents = list(set([dict_['entity'] for sentence in self.df.entities for dict_ in sentence]))
        return self.unique_ents

        
    def get_unique_intents(self):
        self.unique_intents = list(set(self.df.intent))
        return self.unique_intents
        
    
    def get_entities(self):
        
        self.clean_and_tokenise()
        
        self.entitites_start_stop = df.entities.apply(lambda x: [(w['start'], w['stop']) for w in x])
    
        self.entity_sequence = list(map(lambda x: ['O' for word in x], self.sentences_tokenised))
        

        for i, sent in enumerate(self.sentences_tokenised):
            for ent in self.df.iloc[i].entities:
                start = ent['start']
                stop = ent['stop']
                
                entity = ent['entity']
                                
                
                self.entity_sequence[i][start:stop+1] = [entity for i in range(len(self.entity_sequence[i][start:stop+1]))]
#                except IndexError:
#                    start -= 1
#                    stop -= 1
#                    
#                    self.entity_sequence[i][start:stop+1] = [entity for i in len(self.entity_sequence[i][start:stop+1])]


        return self.entity_sequence
        
        
        
data = DataGetter('ChatbotCorpus.json')

df = data.df

unique_entities = data.get_unique_entities()

sentences = data.clean_and_tokenise()

entities = data.get_entities()

entities_start_stop = df.entities.apply(lambda x: [(w['start'], w['stop']) for w in x])

unique_intents = data.get_unique_intents()





entities = list(map(lambda x: ['O' for word in x], sentences))

#entities = []

faulty_points = []


for i, sent in enumerate(sentences):
    for ent in df.iloc[i].entities:
        start = ent['start']
        stop = ent['stop']
        text = ent['text']
                
        entity = ent['entity']
        
        text = re.split(" |(-)|'|(:)|(,)", text)
                        
        if not any(set(sentences[i][start:stop+1]) & set(text)):
            faulty_points.append(i)
            break
            
        len_entity = len(entities[i][start:stop+1])
        newlist = [entity] * len_entity
        entities[i][start:stop+1] = newlist
            
        



for i in faulty_points:
    
    for ent in df.iloc[i].entities:
        
        start = ent['start'] - 1
        stop = ent['stop'] - 1
        
        entity = ent['entity']
        
        sentences[i][start:stop+1]
        
        len_entity = len(entities[i][start:stop+1])
        
        newlist = [entity] * len_entity
        entities[i][start:stop+1] = newlist
    
    
    


sent_ents = []
    
for i, sent in enumerate(sentences):
    
    word_ent = []
    for word, ent in zip(sent, entities[i]):
        word_ent.append((word, ent))
    
    sent_ents.append(word_ent)
        
   


corpus = []


for sent in df.entities:
    #sent = sent[0]
    
    for dict_ in sent:
        word = dict_['text']
        ent = dict_['entity']
    corpus.append((word, ent))
    
corpus_df = pd.DataFrame(corpus, columns=["word", "ent"])


ent_plot = sns.countplot(corpus_df['ent'])
plt.xticks(rotation=45)

plt.figure(figsize=(40, 6));
word_plot = sns.countplot(corpus_df['word'], order=corpus_df['word'].value_counts().index)
plt.xticks(rotation=60);





def word2features_train(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
        })
    else:
        features['EOS'] = True

    return features


def word2features_test(sent, i):
    
    
    word = sent[i][0]
  

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent, train):
    if train == True:
        return [word2features_train(sent, i) for i in range(len(sent))]
    
    else:
        return [word2features_test(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

train_index = df[df.training == True].index.tolist()
test_index = df[df.training == False].index.tolist()


sent_ents_train = [sent_ents[i] for i in train_index]

sent_ents_test = [sent_ents[i] for i in test_index]


X_train = [sent2features(s, train=True) for s in sent_ents_train]
y_train = [sent2labels(s) for s in sent_ents_train]


X_test = [sent2features(s, train=False) for s in sent_ents_test]
y_test = [sent2labels(s) for s in sent_ents_test]




# Random Forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, f1_score, classification_report


X_train_rf = pd.DataFrame()

for sent in X_train:
    df_rows = pd.DataFrame(sent)
    
    X_train_rf = pd.concat([X_train_rf, df_rows])
    
X_train_rf.reset_index(drop=True, inplace=True)





X_test_rf = pd.DataFrame()

for sent in X_test:
    df_rows = pd.DataFrame(sent)
    
    X_test_rf = pd.concat([X_test_rf, df_rows])
    
X_test_rf.reset_index(drop=True, inplace=True)



y_train_rf = [label for sent in y_train for label in sent]

y_test_rf = [label for sent in y_test for label in sent]



for col in X_train_rf.columns:
    if X_train_rf[col].dtype == 'object':
        
        X_train_rf[col] = X_train_rf[col].astype('category')
        
        X_test_rf[col] = X_test_rf[col].astype('category')
        
        col_ohe_train = pd.get_dummies(X_train_rf[col], drop_first=True, prefix=col)
        col_ohe_test = pd.get_dummies(X_test_rf[col], drop_first=True, prefix=col)
        
        del X_train_rf[col]
        del X_test_rf[col]
        
        X_train_rf = pd.concat([X_train_rf, col_ohe_train], axis=1)
        
        X_test_rf = pd.concat([X_test_rf, col_ohe_test], axis=1)




X_train_rf, X_test_rf = X_train_rf.align(X_test_rf, join='inner', axis=1)





rf = RandomForestClassifier(max_depth=20, random_state=0,
                           n_estimators=100, oob_score=True,
                           n_jobs=-1, verbose=0,
                           max_features='auto')

rf.fit(X_train_rf, y_train_rf)


pred_rf = rf.predict(X_test_rf)

#for i, j in zip(y_test_rf, pred_rf):
#    print(i, j)


print(classification_report(y_test_rf, pred_rf))

accuracy_score(y_test_rf, pred_rf)

f1_score(y_test_rf, pred_rf, average='weighted')

#pred = cross_val_predict(RandomForestClassifier(n_estimators=200),
#                         X=X_train_rf, y=y_train_rf, cv=5)






from sklearn_crfsuite import CRF

from sklearn_crfsuite.metrics import flat_f1_score

%%time
crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=True)



from sklearn_crfsuite.metrics import flat_classification_report


crf.fit(X_train, y_train)


pred = crf.predict(X_test)

#pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y_test)

print(report)

flat_f1_score(y_test, pred, average='weighted')



for i, j in zip(pred, test_index):
    print(i, "\n", sentences[j], "\n"*2)

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
        


intent_plot = sns.countplot(df.intent)

df_intent = df.loc[: ,['text', 'intent', 'entities', 'training']]


ent_count = {}


for i, sent in enumerate(df.intent):
    
    ent_sent = []
    
    for dict_ in sent:
    
        dict_['entity']





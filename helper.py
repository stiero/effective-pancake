#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:40:42 2019

@author: msr
"""

def word2features_train(sent, i):
    
    """ Takes a list of sentences and generates features for every word in it 
    - for a training dataset
    
    Args: 
        sent -> a list of sentences containing the (word, entity) tuple
        i -> the index of the current word
            
    Returns: 
        features -> a dictionary of features for word at index i
    
    Raises:
        Fails if sent or i aren't in the expected format
    """
    
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),           
        'word[-3:]': word[-3:],                         # Substring of word
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),               # Flags if condition is met
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        nertag1 = sent[i-1][1]                           # NER tag of previous word
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:nertag': nertag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        nertag1 = sent[i+1][1]                               # NER tag of next word
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': nertag1,
        })
    else:
        features['EOS'] = True

    return features


def word2features_test(sent, i):
    
    """ Takes a list of sentences and generates features for every word in it 
    - for a test dataset
    
    Args: 
        sent -> a list of sentences containing the (word, entity) tuple
        i -> the index of the current word
            
    Returns: 
        features -> a dictionary of features for word at index i
    
    Raises:
        Fails if sent or i aren't in the expected format
    """
    
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
    
    """ Takes a 'sent' object and generates features for each word in it
    
    Args:
        sent -> a list of sentences containing the (word, entity) tuple
        train -> True/False whether or not 'sent' is from the training data
    
    Returns:
        list of the outputs of word2features for a given 'sent' object
        
    Raises:
        Fails if 'sent' does not match the expected format
    """
    
    if train == True:
        return [word2features_train(sent, i) for i in range(len(sent))]
    else:
        return [word2features_test(sent, i) for i in range(len(sent))]

    
def sent2labels(sent):
    
    """ Takes a 'sent' object and extracts entity label for each word in it
    
    Args:
        sent -> a list of sentences containing the (word, entity) tuple
        
    Returns:
        list of the entity labels for each word in a 'sent' object
        
    Raises:
        Fails if 'sent' does not match the expected format
    """
    
    return [label for token, label in sent]







def sample_ent_test_df(sample_text_list, model):
    
    """ Given a list input, formats and generates an NER prediction 
    for it using our trained model.
    
    Args:
        sample_text -> String of a sample sentence
        
    Returns:
        pred_crf -> NER prediction from our CRF model
        sample_text -> Processed string from the input
        
    Raises:
        Fails if sample_text is not a list
    """
    
    sample_text_list = [[(word, None) for word in sent] for sent in sample_text_list]
    
    preds = []
    
    for sent in sample_text_list:
    
        X_test_sample = [sent2features(sent, train=False)]
        pred_crf = model.predict(X_test_sample)
        
        preds.append(pred_crf)
    
    return preds, sample_text_list
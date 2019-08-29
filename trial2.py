#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:57:40 2019

@author: msr
"""

def common_member(a, b): 
    
    a_set = [elem for sublist in a for elem in sublist]
    
    b_set = [elem for sublist in b for elem in sublist]
    a_set = set(a_set) 
    b_set = set(b_set) 
    if (a_set & b_set): 
        print(a_set & b_set) 
    else: 
        print("No common elements")  
        
        
common_member(X_train, X_test)




result = set(X_train)
for s in p[1:]:
    result.intersection_update(s)
print(result


X_train_flattened = [elem for sublist in X_train for elem in sublist]

X_test_flattened = [elem for sublist in X_test for elem in sublist]


for i, train_row in enumerate(X_train):
    for j, test_row in enumerate(X_test):
        
        if train_row == test_row:
            print((i, j))
            
            
            62 101
            
df_train = df[df.training == True]

df_test = df[df.training == False]





sample = "Can you tell me the shortest route from Whitefield to Majestic?"

sample = re.sub('[.!]+', "", sample)

sample = re.split(" |(-)|'|(:)|(,)|(\?)", sample)


sample = [(word, None) for word in sample if word != None]



X_test_sample = [sent2features(sample, train=False)]


sample_tags = ['O', 'O', 'O', 'O', 'O', 'Criterion', 'O', 'StationStart', 'O', 'StationDest']


pred = crf.predict(X_test_sample)

#pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)

report = flat_classification_report(y_pred=pred, y_true=y_test)

print(report)

flat_f1_score(y_test, pred, average='weighted')








sample_intent = "Can you tell me the quickest way to go from Bangalore to London?"


sample_intent = re.sub('[.!]+', "", sample_intent)

sample_intent = re.split(" |(-)|'|(:)|(,)|(\?)", sample_intent)


sample_intent = [word for word in sample_intent if word != None]


#X_sample_intent = np.zeros((1, max_len))

X_sample_intent = np.zeros((1, len(sample_intent)))


for i, word in enumerate(sample_intent):
    
    if word not in word2idx.keys():
        continue
    
    X_sample_intent[:,i] = word2idx[word]
    

X_sample_intent = pad_sequences(maxlen=max_len, sequences=X_sample_intent, 
                               padding="post", value=n_words - 1)    


pred_sample_intent = model.predict(X_sample_intent)

np.argmax(pred_sample_intent)









sample_intent = [word2idx[w] for w in sample_intent]

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 12:15:32 2016

@author: Prudhvi
"""
import os
import nltk
os.chdir("C:\Users\Prudhvi\Desktop\data science\Project")

import string 
import csv 
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score 
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize          


med_data = pd.read_csv('TextClassification_Data.csv')
full_data = list(med_data['DATA'])
target = list(med_data['sub_categories'])

stop = stopwords.words('english')
rm_list = ["phonexxxx-xxxx","for:","additional","Phone","Note", "call", "patient", "back", "at:", "cell", "phone", "caller", "from", "patient", "caller:", "name:", "other", "taken:", "by:", "Follow-up","Details:","default" ,"Action","Taken","Default","#","%","&","'","''","'*","'+","'-","'..","'-thomas","'..next","'.please","'/","'="]

stop = stop + rm_list
stemmer = PorterStemmer()
final_data = [] 
############# pre processing for the 'Data' column

for j in range(len(full_data)) :
    split = full_data[j].split(' ')
    regex = re.compile('\s+')
    new = []
    tokens =[]
    for i in range(len(split)) :
        if len(regex.findall(split[i])) == 0 and (re.search( r'\\|\x080|\d|xxxx-xxxx| |//|#|&|\|,' ,split[i], re.M|re.I)) == None :
            
            new.append(str(split[i].lower()))
            ##### vocab.append(split[i].lower())
            
    tokens = [token for token in new if token not in stop]
    ### new_tokens = [correction(token) for token in tokens]
    ### final_tokens = [stemmer.stem(token) for token in new_tokens]
    final_data.append(str(list(tokens)))


spells = [token for token in final_names if len(wordnet.synsets(token)) > 0 ] 
spells = list(set(spells))
mistakes = [token for token in final_names if len(wordnet.synsets(token)) == 0 ] 


""""with open('mistakes.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in mistakes :
        writer.writerow([val])"""

d = {}

for word in mistakes:
    k = correction(word)
    if k != word : 
        d[word] = k

spell_cor = final_data 

## len(spell_cor)

for lst in range(len(spell_cor)):
    for j in range(len(spell_cor[lst])) :
        if spell_cor[lst][j] in d.keys():
            spell_cor[lst][j] = d[spell_cor[lst][j]]
    

#### vocab_final = list(set(vocab)) 
####final_data = [lst for lst in final_data if len(lst) > 0]

stem_words = []

for lst in range(len(spell_cor)):
    stemmed = []
    for j in range(len(spell_cor[lst])) :
        stemmed.append((stemmer.stem(spell_cor[lst][j])))
    stem_words.append(list(stemmed))


stem_final = []
for i in range(len(stem_words)):
    stem_final.append(str(stem_words[i]))

        
###### feature_names = [stemmer.stem(token) for token in feature_names]
spell_final = []
for i in range(len(spell_cor)):
    spell_final.append(str(spell_cor[i]))
    
    
count_data = final_data 
final_names = []
##### counting the most frequent words for each category
 
cat_names = med_data.sub_categories.unique()

for i in range(len(cat_names)):
    sample = []
    for j in range(len(count_data)):
        if target[j] == cat_names[i]:
            sample.append((count_data[j]))
    count_vect = CountVectorizer(analyzer='word',strip_accents= 'ascii', 
                                 max_features = 400)
    each_cat = count_vect.fit_transform(sample)
    each_cat_words = count_vect.get_feature_names()
    final_names.append(each_cat_words)

final_names = [item for sublist in final_names for item in sublist] 
final_names = list(set(final_names))
   

count_vect = CountVectorizer(analyzer='word', strip_accents= 'ascii' , max_features = 500) 
count_td = count_vect.fit_transform(count_data) 
count_fnames = count_vect.get_feature_names()
    



vect = TfidfVectorizer(analyzer='word', strip_accents= 'ascii' , vocabulary = final_names) 
tfidf = vect.fit_transform(spell_new) 


feature_names = vect.get_feature_names() 
new_features = list(set(feature_names))
new_features.sort()


dense = tfidf.todense() 
dense.shape 
df = pd.DataFrame(dense)
df['target'] = target


    
import random
import numpy as np 
random.seed(800)
from sklearn.cross_validation import train_test_split

df = df.iloc[np.random.permutation(len(df))]
train = df[0:int(len(df)*0.7)]
test = df[int(len(df)*0.7):]


###### multinomial NB 

from sklearn.naive_bayes import MultinomialNB  
model = MultinomialNB()
model.fit(train.ix[:,:-1],train.ix[:,-1])
pred = model.predict(train.ix[:,:-1]) 
train_acc = accuracy_score(train.ix[:,-1],pred) 
pred_test = model.predict(test.ix[:,:-1]) 
test_acc = accuracy_score(test.ix[:,-1],pred_test) 
print train_acc, test_acc



from sklearn import svm 
#### data transformation for svc 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train_svc = le.fit_transform(train.ix[:,-1])
test_svc = le.fit_transform(test.ix[:,-1])
m = svm.LinearSVC() 
m.fit(train.ix[:,:-1],train_svc) 

svm_train_pre = m.predict(train.ix[:,:-1])
train_acc = accuracy_score(train_svc,svm_train_pre)
svm_pre = m.predict(test.ix[:,:-1])
test_acc = accuracy_score(test_svc,svm_pre) 
print(train_acc, test_acc)



##### with stratifying

y = df.pop('target')

y[3000] = 'mEDICATION RELATED'
X_train, X_test, y_train, y_test = train_test_split( df, y, test_size=0.3, stratify = y)


from sklearn import svm 
#### data transformation for svc 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train_svc = le.fit_transform(y_train)
test_svc = le.fit_transform(y_test)
m = svm.LinearSVC() 
m.fit(X_train,train_svc) 

svm_train_pre = m.predict(X_train)
train_acc = accuracy_score(train_svc,svm_train_pre)
svm_pre = m.predict(X_test)
test_acc = accuracy_score(test_svc,svm_pre) 
print(train_acc, test_acc)





""" Neural network not converging
from sknn.mlp import Classifier, Layer help()

nn = Classifier(layers=[Layer("Rectifier", units=100), Layer("Linear")],learning_rate=0.02, n_iter=15)

le = preprocessing.LabelEncoder()
train_nn= train.ix[:,:-1].as_matrix()
test_nn = test.ix[:,:-1].as_matrix()
train_att = train.ix[:,-1]
test_att = test.ix[:,-1]

nn.fit(train_nn,train_att) 


print(test_acc)"""










    
    
\\par\x08

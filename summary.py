# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:38:31 2016

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
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score 
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize     
from sklearn.naive_bayes import MultinomialNB   
import random
import numpy as np     

med_data = pd.read_csv('TextClassification_Data.csv') 

med_data.replace('', np.nan, inplace=True)
med_data = med_data.dropna()
med_data = med_data.reset_index(drop=True)

summ_data = list(med_data['SUMMARY'])  

target = list(med_data['sub_categories']) 

stop = stopwords.words('english')

rm_list = ["phonexxxx-xxxx","for:","additional","Phone","Note", "call", "patient", "back", "at:", "cell", "phone", "caller", "from", "patient", "caller:", "name:", "other", "taken:", "by:", "Follow-up","Details:" ,"Action","Taken","Default","#","%","&","'","''","'*","'+","'-","'..","'-thomas","'..next","'.please","'/","'="] 


stop = stop + rm_list

vocab = [] 
sum_final = []
for j in range(len(summ_data)) :
    sp = str(summ_data[j]).split(' ')
    regex = re.compile('\s+')
    final_tokens = []
    toks =[]
    for i in range(len(sp)) :
        if len(regex.findall(sp[i])) == 0  and (re.search( r'\d' ,sp[i], re.M|re.I)) == None :
            toks.append(sp[i].lower())
            ##### vocab.append(split[i].lower())
            
    final_tokens = [tok for tok in toks if tok not in stop]
    ### new_tokens = [correction(token) for token in tokens]
    ### final_tokens = [stemmer.stem(token) for token in new_tokens]
    sum_final.append(str(list(final_tokens)))


sum_vect = TfidfVectorizer(analyzer='word', strip_accents= 'ascii' , max_features = 15000)
 
sum_tfidf = sum_vect.fit_transform(sum_final) 

sum_dense = sum_tfidf.todense() 
sum_dense.shape 
sum_features = sum_vect.get_feature_names()
sum_df = pd.DataFrame(sum_dense)

sum_df['target'] = target

        
    


sum_df = sum_df.iloc[np.random.permutation(len(sum_df))]


sum_train = sum_df[0:int(len(sum_df)*0.7)]
sum_test = sum_df[int(len(sum_df)*0.7):]



###### multinomial NB 



sum_model = MultinomialNB()

sum_model.fit(sum_train.ix[:,:-1],sum_train.ix[:,-1])
pred = sum_model.predict(train.ix[:,:-1]) 
train_acc = accuracy_score(train.ix[:,-1],pred) 

pred_test = sum_model.predict(test.ix[:,:-1]) 
test_acc = accuracy_score(test.ix[:,-1],pred_test) 

print train_acc, test_acc
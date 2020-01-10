# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:46:41 2016

@author: Syedghouri
"""

import os
import re
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

os.chdir("C:\Users\Prudhvi\Desktop\data science\Project") 

import pandas as pd
import numpy as np
from cStringIO import StringIO
from pyth.plugins.rtf15.reader import Rtf15Reader
from pyth.plugins.plaintext.writer import PlaintextWriter
from pyth.plugins.python.reader import PythonReader

#reading the data
given_data = pd.read_csv("TextClassification_Data.csv", delimiter = ",")

#To know column names
#given_data.columns.values

#given_data["new_cat"] = given_data["categories"] + "_" + given_data["sub_categories"]
#given_data["new_cat"]

#droping the rows having empty values. Reseting the index after droping
given_data.replace('', np.nan, inplace=True)
given_data = given_data.dropna()
given_data = given_data.reset_index(drop=True)

#short_data(StringIO(given_data['DATA'][53931]))

#given_data2 = given_data
#len(given_data2)
#given_data2 = given_data2.drop(2)
#given_data2 = given_data2.reset_index(drop=True)
#len(given_data2)'''
#for i in range(1,len(given_data)):
 #   given_data['new_feature'][i] = short_data(StringIO(given_data['DATA'][i]))

#given_data['new_feature'] = given_data.apply(lambda row: short_data(row),axis=1)

parsed_data = []

len(given_data)
len(parsed_data)

#the following loop ceates a list of cleaned data for each row. Also it removes the data which is not compatable with 
#rtf format. There are five records of such time anyways. The following are their indexes 
#9584, 10372, 17990, 36301, 46985. I have removed these five and finally the index is reset.
#So the length of given_data now is 53927 and parsed data is also a list of length 53927. Hence we can create
#a new column of parsed_data and use that as out complete data.

for i in range(0,len(given_data)):
    
    str1 = given_data["DATA"][i]
    
    try:
        
        parsed_data.append(short_data(str1))
    except TypeError:
        given_data = given_data.drop(i)
    finally:
        given_data = given_data.reset_index(drop=True)
 
#both length are same.       
#len(given_data)
#len(parsed_data)

#adds a new colum by name new_feature which has cleaned text from data column. now we can use this as our
#complete data for classification
given_data["new_feature"] = parsed_data


#x = given_data["DATA"][10371]


print short_data(x)

def short_data(x1):
    
    try:
        
        x = StringIO(x1)
        doc = Rtf15Reader.read(x)
    
        ans = PlaintextWriter.write(doc).getvalue()
        #print ans
    
        my_str = ans.replace('\n', ' ')
    
        rm_list = ["phonexxxx-xxxx","for:","additional","Phone","Note", "call", "patient", "back", "at:", "cell", "phone", "caller", "from", "patient", "caller:", "name:", "other", "taken:", "by:", "Follow-up","Details:" ,"Action","Taken"]
    
        rm_list[0]
    
        words_list = my_str.lower().split()
    
        return ' '.join([i for i in words_list if i not in (x.lower() for x in rm_list)])
        
    
    except Exception as ex:
        
        #print ex
        concat_str = "{\\rtf1\\ansi\\ftnbj{\\fonttbl{" + x1
        
        clean_str = concat_str.replace("xxxx-xxxx"," ")
        
        #print clean_str
        
        x = StringIO(clean_str)
        doc = Rtf15Reader.read(x)
    
        ans = PlaintextWriter.write(doc).getvalue()
        #print ans
    
        my_str = ans.replace('\n', ' ')
    
        rm_list = ["phonexxxx-xxxx","for:","additional","Phone","Note", "call", "patient", "back", "at:", "cell", "phone", "caller", "from", "patient", "caller:", "name:", "other", "taken:", "by:", "Follow-up","Details:" ,"Action","Taken"]
    
        rm_list[0]
    
        words_list = my_str.lower().split()
    
        return ' '.join([i for i in words_list if i not in (x.lower() for x in rm_list)])
        
    '''except Exception as Argument:
        print Argument
        
        concat_str = "{\\rtf1\\ansi\\ftnbj{\\fonttbl{" + x1['DATA']
        
        x = StringIO(concat_str)
        doc = Rtf15Reader.read(x)
    
        ans = PlaintextWriter.write(doc).getvalue()
        #print ans
    
        my_str = ans.replace('\n', ' ')
    
        rm_list = ["xxxx-xxxx","phonexxxx-xxxx","for:","additional","Phone","Note", "call", "patient", "back", "at:", "cell", "phone", "caller", "from", "patient", "caller:", "name:", "other", "taken:", "by:", "Follow-up","Details:" ,"Action","Taken"]
    
        rm_list[0]
    
        words_list = my_str.lower().split()
    
        return ' '.join([i for i in words_list if i not in (x.lower() for x in rm_list)])'''
        
        
    
        
        


'''given_data.iloc[0]
content101 = StringIO(r"{\rtf1\ansi\ftnbj{\fonttbl{\f0 \fswiss Arial;}}{\colortbl ;\red255\green255\blue255 ;\red0\green0\blue255 ;\red0\green0\blue0 ;\red0\green0\blue255 ;\red0\green128\blue0 ;}{\stylesheet{\f0\fs20\cf3\cb1 Normal;}{\cs1\additive\cf3\cb1 Default Paragraph Font;}}\margl1440\margr1440\margt540\margb1440\headery540\footery720\formshade\sectd\marglsxn1440\margrsxn1440\margtsxn540\margbsxn1440\headery540\footery720\sbkpage\pgncont\plain\plain\fs20\pard\plain\fs20\cf0\fs24\sscharaux1\b Phone Note \fs20\b0\par\b\par Call patient back at:\par\b0 Home Phonexxxx-xxxx\par\fs24\b\par CALL FROM PATIENT\par\fs20 Caller: \b0 Patient\par\par\fs24\b Prescription \fs20\b0\par\b Other Prescription Comments:\par\b0 pt called to discuss nortryptiline. she says she has a weird taste in her mouth and pt also sweats heavily, please call to discuss.\par\b Call Taken by: \b0 xxxx-xxxx ,  May 22, 2015 9:09 AM\par\par\fs24\b RNA Follow-up\fs20\b0\par\b Follow-up Details: \b0 pt states her food has no favor in last 2 week Started Nortriptyline 25 mg4/23  and she is having problem with excessive sweating and pain is unrelieved. Please advise next steep. \par\b Follow-up by: \b0 Mary Morton LPN ,  May 22, 2015 9:40 AM\par\par\fs24\b RNA Follow-up\fs20\b0\par\b Additional Follow-up Details: \b0 ok to stop\par\b Additional Follow-up by: \b0xxxx-xxxx MD,  May 22, 2015 11:03 AM\par\plain\fs20\par\cf3\par\plain\fs20\par}")
print short_data(given_data.iloc[0])

strr =  "{\\rtf1\\ansi\\ftnbj{\\fonttbl{" + given_data.iloc[610]['DATA']
print strr

x1=given_data.iloc[9584]
x = "{\\rtf1\\ansi\\ftnbj{\\fonttbl{" + given_data.iloc[9584]['DATA']
x = StringIO(x)'''

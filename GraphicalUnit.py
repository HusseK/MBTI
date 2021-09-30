# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:03:45 2021

@author: hkeita
"""

import matplotlib.pyplot as plt
import seaborn as sns
from collections import  Counter
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import nltk



class GraphicalUnit():
    
    def __init__(self,MBTI):
        self.MBTI = MBTI
        
        
        
    def plot_top_stopwords_barchart(self,text):
        stop=set(stopwords.words('english'))

        new= text.str.split()
        new=new.values.tolist()
        corpus=[word for i in new for word in i]
        from collections import defaultdict
        dic=defaultdict(int)
        for word in corpus:
            if word in stop:
                dic[word]+=1

        top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:20] 
        x,y=zip(*top)
        plt.figure(figsize=(12,12))
        plt.title("Occurence des stopwords dans le texte")
        plt.bar(x,y)
        
        
        
        
    def plot_top_non_stopwords_barchart(self,text):
        stop=set(stopwords.words('english'))
    
        new= text.str.split()
        new=new.values.tolist()
        corpus=[word for i in new for word in i]
    
        counter=Counter(corpus)
        most=counter.most_common()
        x, y=[], []
        for word,count in most[:40]:
            if (word not in stop):
                x.append(word)
                y.append(count)
    
        sns.barplot(x=y,y=x)
        plt.title("Occurence des mots (stopwords non compris)")
        
    def plot_top_ngrams_barchart(text, n=2):
        stop=set(stopwords.words('english'))
    
        new= text.str.split()
        new=new.values.tolist()
        corpus=[word for i in new for word in i]
    
        def _get_top_ngram(corpus, n=None):
            vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0) 
            words_freq = [(word, sum_words[0, idx]) 
                          for word, idx in vec.vocabulary_.items()]
            words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
            return words_freq[:10]
    
        top_n_bigrams=_get_top_ngram(text,n)[:10]
        x,y=map(list,zip(*top_n_bigrams))
        sns.barplot(x=y,y=x)
        
        
    
        
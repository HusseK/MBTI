import pandas as pd 
import unicodedata
import string
import stanza
import nltk
import re
import pickle
import numpy as np
from textblob import TextBlob

from gensim.parsing.preprocessing import remove_stopwords

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize
from nltk.tag import StanfordPOSTagger
from autocorrect import Speller




class DataUnit():
    def __init__(self, MBTI):
        self.pdata=None
        self.data=pd.read_csv("mbti_1.csv")
        try:
            with open('processed_data.pkl', 'rb') as f:
                self.pdata = pickle.load(f)
        except:
            print("Je n'ai pas réussi à charger les données. Je vais devoir les pré traiter.")
        #stanza.download('en', package='craft')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.MBTI= MBTI
        


        
    def population(self,carac):
        return self.data.loc[data['type']==carac]
    
    
    def strip_accents(self,s):
        for i in range(self.data.shape[0]):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn') 
        
        
    def tolower_text(self,s):
        for i in range(self.data.shape[0]):
             return s.lower()
            
    def correct_typos(self, s):
        spell = Speller(lang='en')
        return spell(s)

    def correct_typos2(self, s):
        a= TextBlob(s)
        return (a.correct())
            
    def tokenization(self,s):
        sequences = sent_tokenize(s)
        return[word_tokenize(seq) for seq in sequences]
        
        for i in range(self.data.shape[0]):
            text_tokens = word_tokenize(self.self.data.iloc[i,1])
                 
    def remove_url(self,s):
        for i in range(self.data.shape[0]):
            return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', s) 
        
    def remove_punctuation(self, s):
        return s.translate(str.maketrans('', '', string.punctuation))
    
    def remove_punctuation_tokens(self, seq_tokens):
        no_punct_seq_tokens = []
        for seq_token in seq_tokens:
                no_punct_seq_tokens.append([token for token in seq_token if token not in string.punctuation])
        return no_punct_seq_tokens
            
    def PosTAG(self,s):
        nlp = stanza.Pipeline(lang='en', processors='tokenize, pos')
        doc=nlp(s)
        print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')
        
    def pos_tag(self,tokens):
        
        return(nltk.pos_tag(tokens))
    
    def remove_numbers(self,s):
        return re.sub(r'\d+', '', s)
        
            
    def lematize(self,s, pos=''):
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize(s, pos=pos)
        
        

    def get_wordnet_pos(self,s):
    
        if s.startswith('J'):
            return wordnet.ADJ
        elif s.startswith('V'):
            return wordnet.VERB
        elif s.startswith('N'):
            return wordnet.NOUN
        elif s.startswith('I'):
            return wordnet.NOUN
        elif s.startswith('R'):
            return wordnet.ADV
        else:
            return None
        
        
    def preprocess(self):
        for i in range(self.data.shape[0]):
            self.data.iloc[i,1] = self.remove_url(self.data.iloc[i,1])
            self.data.iloc[i,1] = self.remove_numbers(self.data.iloc[i,1])
            self.data.iloc[i,1] = self.tolower_text(self.data.iloc[i,1])
            self.data.iloc[i,1] = self.strip_accents(self.data.iloc[i,1])
            self.data.iloc[i,1] = remove_stopwords(self.data.iloc[i,1])
            self.data.iloc[i,1] = self.tokenization(self.data.iloc[i,1])
            #self.data.iloc[i,1] = self.data.iloc[i,1].split('|||')
            #self.data.iloc[i,1]=' '.join(self.data.iloc[i,1])
    
            #self.data.iloc[i,1] = self.remove_punctuation(self.data.iloc[i,1])
            with open('processed_data__.pkl', 'wb') as f1:
                    pickle.dump(self.data, f1)
            
            
            
        ##############################
        ##Lematization avec Pos Tag ##
        ##############################
    def lematize_with_pos(self):
        
        for i in range(self.data.shape[0]): #pour chaque individu
            #for j in range(len(self.data.iloc[i,1])): #pour chaque phrase
            _pos = np.array(nltk.pos_tag(self.data.iloc[i,1][0]))[:,1]
            for k in range(len(self.data.iloc[i,1][0])): #pour chaque mot
                #print("#####################")
                #print(k)
                #print(_pos[k])
                #print(self.data.iloc[i,1][0][k])
                #print("#####################")
                #self.data.iloc[i,1][0][k]= self.correct_typos(self.data.iloc[i,1][0][k])
                if _pos[k] is None:
                    continue
                try:
                    
                    self.data.iloc[i,1][0][k] = self.lematize(self.data.iloc[i,1][0][k], pos=self.get_wordnet_pos( _pos[k]))
                except:
                    0
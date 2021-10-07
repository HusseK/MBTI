import pandas as pd
import tensorflow as tf 
import unicodedata
import string
import stanza
import nltk
import re
import pickle
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.parsing.preprocessing import remove_stopwords
from transformers import BertTokenizer
import contractions
from contractions import fix

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import sent_tokenize, word_tokenize
from nltk.tag import StanfordPOSTagger
from autocorrect import Speller
from gensim.models import word2vec
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score




class DataUnit():
    def __init__(self, MBTI):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokens=None
        self.attention_masks = None
        self.pdata=None
        self.data=pd.read_csv("mbti_1.csv")
        self.test_set=None
        self.max_len=0
        try:
            with open('processed_data__.pkl', 'rb') as f:
                self.pdata = pickle.load(f)
        except:
            print("Je n'ai pas réussi à charger les données. Je vais devoir les pré traiter.")
        #stanza.download('en', package='craft')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.MBTI= MBTI
        
        
        
    def prepare_data(self):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.data.iloc[:,0])
        for i in range(self.data.shape[0]):
            self.data.iloc[i,0]=integer_encoded[i]
        
        train, test = train_test_split(self.data, test_size=0.2, random_state=1)
        X_train = train['posts']
        X_test = test['posts']
        y_train = train['type']
        y_test = test['type']
        self.data = pd.concat((y_train,X_train),axis=1)
        self.test_set = pd.concat((y_test,X_test),axis=1)
        
        #sm = SMOTE(random_state=2)
        #X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
        
        #return (X_train,y_train), (X_test,y_test)
        

    def one_hot_encode(self, labels):
        return tf.keras.utils.to_categorical(labels, num_classes=16)
       
 
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
    
    def remove_punctuation_in_tokens(self, tokenzzz):
        res=[]
        for i in range(len(tokenzzz)):
            if tokenzzz[i] not in string.punctuation:
                res.append(tokenzzz[i])
        return res
        
            
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
        
    def tokenize_sentences(self, sentences, tokenizer, max_seq_len = 500):
        tokenized_sentences = []
    
        for sentence in sentences:
            tokenized_sentence = tokenizer.encode(
                                sentence,                  # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_seq_len,  # Truncate all sentences.
                        )
            
            tokenized_sentences.append(tokenized_sentence)
            
        return tokenized_sentences
    
    def get_max_len(self, sentences):
        max_len=0
        for sent in sentences:
            if len(sent.split())>max_len:
                max_len = len(sent.split())
        return max_len
    
    def create_attention_masks(self,tokenized_and_padded_sentences):
        attention_masks = []
    
        for sentence in tokenized_and_padded_sentences:
            att_mask = [int(token_id > 0) for token_id in sentence]
            attention_masks.append(att_mask)
    
        return np.asarray(attention_masks)
        
        
    def preprocess(self):
        for i in range(self.data.shape[0]):
            self.data.iloc[i,1] = self.remove_url(self.data.iloc[i,1])
            self.data.iloc[i,1] = self.remove_numbers(self.data.iloc[i,1])
            self.data.iloc[i,1] = fix(fix(self.data.iloc[i,1]))
            self.data.iloc[i,1] = self.tolower_text(self.data.iloc[i,1])
            self.data.iloc[i,1] = self.strip_accents(self.data.iloc[i,1])
            self.data.iloc[i,1] = remove_stopwords(self.data.iloc[i,1])
    #    with open('processed_data__.pkl', 'wb') as f1:
    #        pickle.dump(self.data, f1)

    def tokenize_and_create_attention_masks(self):            
        max_seq_len=(self.get_max_len(self.data['posts'].values))
        try:
            with open('tokens_padded.pkl', 'rb') as f:
                self.tokens = pickle.load(f)
        except:
            print("Je n'ai pas réussi à charger les token padded. Je vais essayer de charger les tokens avant padding")
            try:
                with open('tokens.pkl', 'rb') as f:
                    self.tokens = pickle.load(f)
            except:
                print("Je n'ai pas réussi à charger les token. Je vais devoir les regénerer.")    
                self.tokens = self.tokenize_sentences(self.data['posts'].values, self.tokenizer, max_seq_len=max_seq_len)
                with open('tokens.pkl', 'wb') as f1:
                    pickle.dump(self.tokens, f1)
            self.tokens = pad_sequences(self.tokens, maxlen=max_seq_len, dtype="long", value=0, truncating="post", padding="post")
            with open('tokens_padded.pkl', 'wb') as f2:
                pickle.dump(self.tokens, f2)
        
        
        try:
            with open('attention_masks.pkl', 'rb') as f:
                self.tokens = pickle.load(f)
        except:
            print("Je n'ai pas réussi à charger les attention masks. Je vais devoir les regénerer.")
            self.attention_masks = self.create_attention_masks(self.tokens)
            with open('attention_masks.pkl', 'wb') as f3:
                pickle.dump(self.attention_masks, f3)
            
            #input_ids = self.tokenizer.encode(self.data.iloc[i,1], add_special_tokens=True)

            # Update the maximum sentence length.
            #self.max_len = max(self.max_len, len(input_ids))
            
            
            #self.data.iloc[i,1] = self.tokenization(self.data.iloc[i,1])
            #self.data.iloc[i,1] = self.remove_punctuation_in_tokens(self.data.iloc[i,1])
            
            
            #self.data.iloc[i,1] = self.remove_punctuation(self.data.iloc[i,1])
            
            #self.data.iloc[i,1] = self.data.iloc[i,1].split('|||')
            #self.data.iloc[i,1]=' '.join(self.data.iloc[i,1])
    
        #with open('processed_data__.pkl', 'wb') as f1:
        #    pickle.dump(self.data, f1)            
            
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
                    
    
        #word2vec = Word2Vec(all_words, min_count=2)
        
        #model = word2vec.Word2Vec(s, size=300, window=20, min_count=2, workers=1, iter=100)
        
        #vectorizer = CountVectorizer(
        #analyzer = 'word',
        #tokenizer = tokenize,
        #lowercase = True,
        #ngram_range=(1, 1),
        #stop_words = en_stopwords)
    
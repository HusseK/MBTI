# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:31:49 2021

@author: hkeita
"""
import tensorflow as tf
import transformers
import keras.backend as K
import numpy as np
import pickle 
import datetime



class ModelUnit():
    def __init__(self,MBTI):
        self.MBTI=MBTI
        self.DataUnit=MBTI.DataUnit
        self.model= self.create_model()
        self.MAX_LEN = 894 
        self.training_history= None
        self.NB_EPOCHS = 5
        self.BATCH_SIZE = 32
        
        
        
        
        
    def f1_m(self,y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    def recall_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def create_model(self): 
        input_word_ids = tf.keras.layers.Input(shape=(self.MAX_LEN,), dtype=tf.int32,
                                               name="input_word_ids")
        bert_layer = transformers.TFBertModel.from_pretrained('bert-large-uncased')
        bert_outputs = bert_layer(input_word_ids)[0]
        pred = tf.keras.layers.Dense(16, activation='softmax')(bert_outputs[:,0,:])
        
        model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00002), metrics=['accuracy', self.f1_m, self.precision_m, self.recall_m])
        return model
    
    def train_model(self):
        self.training_history=self.model.fit(np.array(self.DataUnit.tokens),
                  self.data.one_hot_labels, verbose = 1, epochs = self.NB_EPOCHS, 
                  batch_size = self.BATCH_SIZE,  
                  callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5)])
        with open(str(datetime.now())+'model.pkl', 'wb') as f2:
                pickle.dump(self.tokens, f2)
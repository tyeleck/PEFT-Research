# -*- coding: utf-8 -*-
"""
Created on Mon May  8 03:13:08 2023

@author: tyele
"""

#%% Setup
import pandas as pd
import keras
import numpy as np
import os

gpus = keras.backend._get_available_gpus()

if len(gpus) == 0:
    print("No GPU's Available")
else:
    print(gpus)


current_path = os.path.dirname(__file__)
current_path = current_path.replace("\\", "/")


#%% Functions

from keras.preprocessing import *
from keras.utils import pad_sequences

import math
import re
import numpy as np



col = {'polarity': 0,
       'tweets': 1
       }



def one_hot_encode(data, dictionary):
    nrow = len(data)
    ncol = len(dictionary)
    
    
    matrix = np.zeros(shape = (nrow,ncol))
    for i in range(nrow):
        for k in range(len(data[i])):
            col = dictionary[data[i][k]]
            matrix[i,col] = 1.0
    return matrix

def df_to_numpy(df):
    df_np = np.array([
        df.iloc[:,0],
        df.iloc[:,1]
    ])
    return df_np

def datasplit(data, ratio = .8):
    data = data.sample(frac = 1)
    
    train_end = math.floor(len(data) * .8)
    train = data[0:train_end]
    test = data[train_end:len(data)]
    
    return train, test
    


def generate_sequences(text_data,#training data
                      maxlen = 50,#maximum length of the embedding sequence
                      max_words = 2000,
                      tokenizer = None):#will only choose consider max_words amount of words for the embedding
    if tokenizer == None:
        tok = text.Tokenizer(num_words=max_words)#Create tokenizer
        tok.fit_on_texts(text_data)#fit tokenizer on texts
    else:
        tok = tokenizer
    
    sequences = tok.texts_to_sequences(text_data) #generate sequences from tokenizer
    
    padded_sequences = pad_sequences(sequences, maxlen = maxlen)#pad sequneces to same length
    
    return (padded_sequences, tok)#return the sequences and the tokenizer

#def kfold(data, model, nfolds = 10):
#    data
    
    
def preprocess(df, maxlen=50, max_words=100, tokenizer=None):
    df_np = df_to_numpy(df)
    y = df_np[0, :].astype('int')
    y = y / 4
    x = df_np[1, :]
    x, tokenizer = generate_sequences(x, maxlen=maxlen, max_words=max_words, tokenizer=tokenizer)
    
    return x, y, tokenizer
def get_accuracy(model):
     fit_model = model.model
     Predictions = fit_model.predict(model.t_x)
     correct = 0
     
     for i in range(len(Predictions)):
         
         prediction = 1 if Predictions[i] > .5 else 0
         
         if(prediction == model.t_y[i]):
             correct += 1
      
     
     accuracy = correct/len(Predictions)
     return accuracy
        

def kfold(data,model, nfolds = 1,  params = {'epochs':5,'batch_size':256}):
    global data_v
    global train
    
    data = data.sample(frac = 1) #Shuffle
    accuracy = list()
    num_obs = len(data)
    
    index_increment = math.floor(num_obs/nfolds)
    
    
    start_index = 0
    
    num_obs = len(data)
    
    data_v = data
    
    accuracys = list()
    for i in range(nfolds):
        
        print("*_____________________________________________________________________*")
        print("Fold: " + str(i+1) + "/" + str(nfolds))
        test_range = range(start_index, start_index + index_increment)
        start_index = start_index + index_increment
       
        
        if(test_range.start == 0):
            train = data[test_range.stop:]
            print("Train Range: " + str(test_range.stop) + " ---> " + str(num_obs))
            
            
            
            
        elif(test_range.stop == num_obs):
            train = data[:test_range.start]
            print("Train Range: " + "0 ---> " + str(test_range.start))
            
            
        else:
            train = pd.concat([data[:test_range.start] , data[test_range.stop:]])
            print("Train Range: " + "0 ---> " + str(test_range.start) + " , " + str(test_range.stop) + " ---> " + str(num_obs))
        print("Test Range: " + str(test_range.start) + " ---> " + str(test_range.stop))
        
        
        test = data[test_range.start:test_range.stop]
    
        model.train(train,test, epochs = params['epochs'],batch_size=params['batch_size'])
        
        accuracy = get_accuracy(model)
        
        print("Test_Accuracy: " + str(accuracy))
    model.result = sum(accuracys) / len(accuracys)
    return model
        
def run_train_test(train, test, model, params = {'epochs':5,'batch_size':256}) :
    model.train(train,test, epochs=params['epochs'],batch_size=params['batch_size'])
    
    fit_model = model.model
    Predictions = fit_model.predict(model.t_x)
    accuracy = get_accuracy(model)
    model.result = accuracy
    print('Test Accuracy: ' + str(accuracy))
    return model
    
    
        
        
#%% Models        
class Model():
    def __init__():
        pass
    def preprocess():
        pass
    def compile_model():
        pass
    def train():
        pass
      
class LSTM_Arch(Model):
    def __init__(self, params = {'max_words': 2000,
                                 'maxlen': 50,
                                 }):
        self.max_words = params['max_words']
        self.maxlen = params['maxlen']
        self.compile_model()
        
    def preprocess(self, train, test):
        self.x,self.y, self.tokenizer = preprocess(train, 
                                    max_words=self.max_words,
                                    maxlen = self.maxlen)
        self.t_x, self.t_y, tokenizer = preprocess(test, 
                                         max_words=self.max_words,
                                         maxlen = self.maxlen,
                                         tokenizer = self.tokenizer
                                         )
    def compile_model(self):
        
        self.model = keras.Sequential([
            layers.Embedding(input_dim = self.max_words,
                                   input_length = self.maxlen,
                                   output_dim = 8),
            layers.LSTM(64, return_sequences = True),
            layers.LSTM(64, return_sequences = False),
            keras.layers.Dense(2, use_bias=True),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])
       
        self.model.compile(
            optimizer = keras.optimizers.RMSprop(learning_rate=0.01),
            loss = 'binary_crossentropy',
            metrics=['accuracy']
        )
    def train(self,train,test,epochs = 5,batch_size = 256):
        self.preprocess(train, test)
        history = self.model.fit(self.x,
                  self.y,
                 batch_size = batch_size,
                 epochs = epochs,
                 verbose=1)
        self.history = history


#%% Run LSTM Model


import tensorflow as tf
#from tensorflow import keras
from tensorflow import keras
from keras import optimizers  



from tensorflow.keras import layers


params = {'max_words': 2000,
          'maxlen': 50}
train_params = {'epochs':1,
                'batch_size':1024}


dataset_dir = current_path + "/Data/train_data.csv"
df = pd.read_csv(dataset_dir, encoding='iso-8859-1', engine = "c").iloc[:,[0,5]]
dataset_dir = current_path + "/Data/test_data.csv"
test_df = pd.read_csv(dataset_dir, encoding='iso-8859-1', engine = "c").iloc[:,[0,5]]



#df = df.iloc[:,1:21]
#df_564 = get_user_movies(df, 564)


#results = kfold(df,model = LSTM_Arch(params), nfolds = 2, params = {'epochs':1,
 #                                                                   'batch_size':1024})
model = run_train_test(df, test_df, model = LSTM_Arch(params), params = train_params)


#%%% PEFT
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType



from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import tensorflow as tf
import tensorflow_hub as hub




model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


module_url = "https://tfhub.dev/bigscience/bloom-560m/1"
model = hub.load(module_url)




#%% 



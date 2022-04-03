# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:23:00 2022

@author: Doug
"""

################
# LIBRARIES
################
import os
import sys

import time
from timeit import default_timer as timer

import pandas as pd
import numpy as np

from sklearn import preprocessing

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPooling2D, AveragePooling2D, LSTM

from keras import backend
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical

import torch
import numba
from numba import jit, cuda

################
# SET TENSORFLOW SESSION TO USE GPU
################
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.keras.backend.get_session()

################
# ENSURE TORCH WORKS
################
print("Is Torch correctly detecting the GPU/CUDA on your system?")
print("Torch Avaiable? %s\nDevice #: %s\nName: %s" % (torch.cuda.is_available(),torch.cuda.current_device(),torch.cuda.get_device_name(0)))

print("Check Tensorflow As Well:")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

###################
# CODE
###################

full_path = 'S:/Applications/Coding/Projects/Machine Learning/Kaggle/prudential-life-insurance-assessment/ST694-Project/Files/Full_Binary_data.csv'
train_path = 'S:/Applications/Coding/Projects/Machine Learning/Kaggle/prudential-life-insurance-assessment/ST694-Project/Files/Train_Binary_data.csv'
test_path = 'S:/Applications/Coding/Projects/Machine Learning/Kaggle/prudential-life-insurance-assessment/ST694-Project/Files/Test_Binary_data.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train_Y = df_train['Response_binary']
df_train_X = df_train.loc[:, df_train.columns != 'Response_binary']

df_test_Y = df_test['Response_binary']
df_test_X = df_test.loc[:, df_test.columns != 'Response_binary']


#Prepare for Hot One Encode (Categories become Integers)
df_train_X = df_train_X.replace('Other',-1)
df_train_X = df_train_X.replace({'A':1,'B':2,'C':3,'D':4,'E':5})

df_test_X = df_test_X.replace('Other',-1)
df_test_X = df_test_X.replace({'A':1,'B':2,'C':3,'D':4,'E':5})



not_float = df_train_X.select_dtypes(exclude=['float']).columns.values.tolist()
not_float.remove('Med_keyword_count') #Do Not Count as Is Poisson Count
is_float = df_train_X.select_dtypes(include=['float']).columns.values.tolist()



df_train_X[not_float] = df_train_X[not_float].astype("category")

df_test_X[not_float] = df_test_X[not_float].astype("category")

#Hot One Encode
df_train_X = pd.get_dummies(df_train_X, columns=not_float)
df_test_X = pd.get_dummies(df_test_X, columns=not_float)


#ds_train_X =  tf.convert_to_tensor(df_train_X)
#ds_train_Y =  tf.convert_to_tensor(df_train_Y)
#ds_test =  tf.convert_to_tensor(df_test)





#with tf.device('/CPU:0'): #Can set GPU or CPU manually - not sure how exactly

#https://www.tensorflow.org/api_docs/python/tf/keras/layers

#Parameters
Learn_Rate = 0.01
batch_size = 100
epochs = 10
steps_per_epoch = len(df_train_X)//batch_size
#validation_steps = len(df_test_X)//batch_size


#Layers
IL_0 = len(df_train_X.columns)
HL_1 = 40
HL_2 = 60
HL_3 = 80
HL_4 = 60
OL_0 = 1 #binary on sigmoid or 2 on softmax

#Model
model = Sequential()

model.add(Input(shape=(IL_0,)))
model.add(Dense(units=HL_1))
model.add(Dense(units=HL_2))
model.add(Dense(units=HL_3))
model.add(Dense(units=HL_4))
model.add(Dense(units=OL_0, activation='sigmoid'))

print(model.summary())

#tf.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','Precision','Recall','AUC'])

history = model.fit(
    df_train_X, df_train_Y,
    batch_size=batch_size,
    epochs=epochs
    #steps_per_epoch=steps_per_epoch
    #validation_data=(df_test_X, df_test_Y),
    #validation_steps=2
)

#history.history #stores values in table form!

scores = model.evaluate(df_test_X, df_test_Y) #need to use full model for appropriate columns
#will need to rewrite code slightly to get working

#Use model.predict(df_test_X) to predict our Y

   
    






    
    
    
    
    
    
    
    
    
    
    
    
    

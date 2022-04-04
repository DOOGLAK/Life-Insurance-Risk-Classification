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

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score

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

full_path = 'S:/Applications/Coding/Projects/Machine Learning/Kaggle/prudential-life-insurance-assessment/ST694-Project/Files/data_final/train_clean_multi.csv'


df_full = pd.read_csv(full_path)
df_full_Y = df_full['Response']
df_full_X = df_full.loc[:, df_full.columns != 'Response']

#Prepare for Hot One Encode (Categories become Integers)
df_full_X = df_full_X.replace({'A':1,'B':2,'C':3,'D':4,'E':5})

not_float = df_full_X.select_dtypes(exclude=['float']).columns.values.tolist()
not_float.remove("Medical_History_1") #Do Not Count as Is Poisson Count
not_float.remove("Medical_History_2") #Do Not Count as Is Poisson Count
already_binary = [s for s in df_full.columns.values.tolist() if "Medical_Keyword" in s]
not_float = [ele for ele in not_float if ele not in already_binary]

#is_float = df_full_X.select_dtypes(include=['float']).columns.values.tolist()


df_full_X[not_float] = df_full_X[not_float].astype("category")


#Hot One Encode
df_full_X = pd.get_dummies(df_full_X, columns=not_float)



#SPLIT DATA
df_train_X, df_test_X, df_train_Y, df_test_Y = train_test_split(df_full_X, df_full_Y, test_size=0.33, random_state=42)


#ds_train_X =  tf.convert_to_tensor(df_train_X)
#ds_train_Y =  tf.convert_to_tensor(df_train_Y)
#ds_test =  tf.convert_to_tensor(df_test)

#ds_Y_test = tf.convert_to_tensor(df_train_Y)

np_new_Y = to_categorical(df_train_Y.to_numpy()-1, num_classes=8)


#with tf.device('/CPU:0'): #Can set GPU or CPU manually - not sure how exactly

#https://www.tensorflow.org/api_docs/python/tf/keras/layers

#Parameters
Learn_Rate = 0.01
batch_size = 32
epochs = 32
steps_per_epoch = len(df_train_X)//batch_size
#validation_steps = len(df_test_X)//batch_size


#Layers
IL_0 = len(df_train_X.columns)
HL_1 = 50
HL_2 = 50
HL_3 = 50
HL_4 = 50
OL_0 = 8 #binary on sigmoid or 2 on softmax

#Model
model = Sequential()

model.add(Input(shape=(IL_0,)))
model.add(Dense(units=HL_1))
model.add(Dense(units=HL_2))
model.add(Dense(units=HL_3))
model.add(Dense(units=HL_4))
model.add(Dense(units=HL_1))
model.add(Dense(units=HL_2))
model.add(Dense(units=HL_3))
model.add(Dense(units=HL_4))
model.add(Dense(units=HL_1))
model.add(Dense(units=HL_2))
model.add(Dense(units=HL_3))
model.add(Dense(units=HL_4))
model.add(Dense(units=OL_0, activation='sigmoid'))

print(model.summary())

#my_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#,'Precision','Recall','AUC'])

history = model.fit(
    df_train_X, np_new_Y,
    batch_size=batch_size,
    epochs=epochs
    #steps_per_epoch=steps_per_epoch
    #validation_data=(df_test_X, df_test_Y),
    #validation_steps=2
)

history #stores values in table form!

#scores = model.evaluate(df_test_X, np_new_Y) #need to use full model for appropriate columns

#will need to rewrite code slightly to get working

#Use model.predict(df_test_X) to predict our Y

my_pred = model.predict(df_test_X)   
my_index=[]

for listy in my_pred:
    my_index.append(listy.tolist().index(max(listy))+1)

my_index

for i in range(0,len(my_index)):
    if my_index[i] <= 6:
        my_index[i] = 0
    else:
        my_index[i] = 1
        
        
df_test_Y = df_test_Y.tolist()

for j in range(0,len(df_test_Y)):
    if df_test_Y[j] <= 6:
        df_test_Y[j] = 0
    else:
        #print('its a 1')
        df_test_Y[j] = 1
    
#my_pred[my_pred<0.5]=0
#my_pred[my_pred>=0.5]=1

print("Accuracy:",accuracy_score(df_test_Y, my_index))
print("Precision:",precision_score(df_test_Y, my_index))
print("Recall:",recall_score(df_test_Y, my_index))

    
    
    
    
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
    
    
    
    

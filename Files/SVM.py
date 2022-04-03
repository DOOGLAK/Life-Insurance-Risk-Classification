# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:03:38 2022

@author: Doug
"""

################
# LIBRARIES
################
import os
import sys

import time
#from timeit import default_timer as timer

import pandas as pd
import numpy as np

#from thundersvm import SVC
from sklearn import svm
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
# ENSURE GPU WORKS
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

##### SVM

temp_x = df_train_X.iloc[0:90000]
temp_y = df_train_Y.iloc[0:90000]

X_train, X_test, Y_train, Y_test = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

# Y_train=Y_train.values.reshape((X_train.shape[0],1))
# Y_test=Y_test.values.reshape((X_test.shape[0],1))

# X_train.shape
# X_test.shape
# Y_train.shape
# Y_test.shape

t0 = time.time()

svm_mod = svm.SVC()
svm_mod.fit(X_train, Y_train)

t1 = time.time()
total = t1-t0
print('In Seconds: %s' % total)
print('In Minutes: %s' % (total/60))
print('In Hours: %s' % (total/60/60))

t2 = time.time()
pred = svm_mod.predict(X_test)
t3 = time.time()
pred_tme = t3-t2
print('In Seconds: %s' % pred_tme)
print('In Minutes: %s' % (pred_tme/60))
print('In Hours: %s' % (pred_tme/60/60))

print("Accuracy:",accuracy_score(Y_test, pred))
print("Precision:",precision_score(Y_test, pred))
print("Recall:",recall_score(Y_test, pred))

#8m for 23,752 rows
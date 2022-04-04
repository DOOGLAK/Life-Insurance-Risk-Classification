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
from sklearn.model_selection import cross_val_score
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
#test_path = 'S:/Applications/Coding/Projects/Machine Learning/Kaggle/prudential-life-insurance-assessment/ST694-Project/Files/Test_Binary_data.csv'
df_train = pd.read_csv(train_path)


df_train_Y = df_train['Response_binary']
df_train_X = df_train.loc[:, df_train.columns != 'Response_binary']


#Prepare for Hot One Encode (Categories become Integers)
df_train_X = df_train_X.replace('Other',-1)
df_train_X = df_train_X.replace({'A':1,'B':2,'C':3,'D':4,'E':5})



not_float = df_train_X.select_dtypes(exclude=['float']).columns.values.tolist()
not_float.remove('Med_keyword_count') #Do Not Count as Is Poisson Count
is_float = df_train_X.select_dtypes(include=['float']).columns.values.tolist()



df_train_X[not_float] = df_train_X[not_float].astype("category")


#Hot One Encode
df_train_X = pd.get_dummies(df_train_X, columns=not_float)







##### SVM
temp_x = df_train_X.iloc[0:10000]
temp_y = df_train_Y.iloc[0:10000]

X_train, X_test, Y_train, Y_test = train_test_split(temp_x, temp_y, test_size=0.2, random_state=42)










#Kernel Options - Linear, poly (needs degree=#), rbf, radial
svm_mod_linear = svm.SVC(kernel='linear', C=1)
svm_mod_linear.fit(X_train, Y_train)
scores = cross_val_score(svm_mod_linear, X_train, Y_train, cv=5)


pred = svm_mod_linear.predict(X_test)

print("Accuracy:",accuracy_score(Y_test, pred))
print("Precision:",precision_score(Y_test, pred))
print("Recall:",recall_score(Y_test, pred))
confusion_matrix(Y_test,pred)







#Kernel Options - Linear, poly (needs degree=#), rbf, radial
svm_mod_poly = svm.SVC(kernel='poly',degree=3)
svm_mod_poly.fit(X_train, Y_train)
scores = cross_val_score(svm_mod_poly, X_train, Y_train, cv=5)

pred = svm_mod_poly.predict(X_test)

print("Accuracy:",accuracy_score(Y_test, pred))
print("Precision:",precision_score(Y_test, pred))
print("Recall:",recall_score(Y_test, pred))
confusion_matrix(Y_test,pred)



#ideal was 0.783, C=3, gamma=default
#Kernel Options - Linear, poly (needs degree=#), rbf, radial
svm_mod_rbf = svm.SVC(kernel='rbf', C=5, gamma=0.02)
svm_mod_rbf.fit(X_train, Y_train)
scores = cross_val_score(svm_mod_rbf, X_train, Y_train, cv=5)

pred = svm_mod_rbf.predict(X_test)

print("Accuracy:",accuracy_score(Y_test, pred))
print("Precision:",precision_score(Y_test, pred))
print("Recall:",recall_score(Y_test, pred))
confusion_matrix(Y_test,pred)









#8m for 23,752 rows
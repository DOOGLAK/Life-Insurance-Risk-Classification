# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 19:05:14 2022

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

import xgboost as xgb
from xgboost import plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, MaxPooling2D, AveragePooling2D
from keras.utils.vis_utils import plot_model

import torch
import numba
from numba import jit, cuda


################
# ENSURE GPU WORKS
################
print("Is Torch correctly detecting the GPU/CUDA on your system?")
print("Torch Avaiable? %s\nDevice #: %s\nName: %s" % (torch.cuda.is_available(),torch.cuda.current_device(),torch.cuda.get_device_name(0)))

print("Check Tensorflow As Well:")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Check XGB:")
print("XGB Available? %s" % bool(xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)))

###################
# CODE
###################

#https://www.datacamp.com/community/tutorials/xgboost-in-python
#https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
#https://xgboost.readthedocs.io/en/stable/parameter.html

train_path = 'S:/Applications/Coding/Projects/Machine Learning/Kaggle/prudential-life-insurance-assessment/ST694-Project/Files/Train_Binary_data.csv'
test_path = 'S:/Applications/Coding/Projects/Machine Learning/Kaggle/prudential-life-insurance-assessment/ST694-Project/Files/Test_Binary_data.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

Y_train = df_train['Response_binary']
X_train = df_train.loc[:, df_train.columns != 'Response_binary']

Y_test = df_test['Response_binary']
X_test = df_test.loc[:, df_test.columns != 'Response_binary']


#Convert to Categorical
cat_list = ['Product_Info_3', 'Employment_Info_2', 'Product_Info_2_char']
X_train[cat_list] = X_train[cat_list].astype("category")
X_test[cat_list] = X_test[cat_list].astype("category")

#XGB Initial
xgb_model = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 1,
                max_depth = 8, alpha = 10, n_estimators = 10, tree_method='gpu_hist',num_parallel_tree=10, enable_categorical=True)

xgb_model.fit(X_train,Y_train)

preds = xgb_model.predict(X_test)
preds_binary = np.round(preds)

accuracy = accuracy_score(Y_test, preds_binary)
accuracy

conf_matrix = confusion_matrix(y_true = Y_test, y_pred = preds_binary)
(conf_matrix[0,0]+conf_matrix[1,1])/np.sum(conf_matrix)

#xgb_model.feature_importances_
#xgb.to_graphviz(xgb_model, num_trees=1)


#xgb.plot_tree(xgb_model,num_trees=1)

#check_importance(xgb_model, X_train)

xgb.plot_importance(xgb_model)
plt.rcParams['figure.figsize'] = [10, 50]
plt.show()


#XGB C-V METHOD
params = {"objective":"reg:logistic",'colsample_bytree': 0.5,'learning_rate': 0.7,
                'max_depth': 8, 'alpha': 10}

df_dmatrix = xgb.DMatrix(data=X_train,label=Y_train, enable_categorical=True)

cv_results = xgb.cv(dtrain=df_dmatrix, params=params, nfold=5,
                    num_boost_round=100,early_stopping_rounds=10,metrics=['auc','error'], as_pandas=True, seed=123)

cv_results.head()
print((1-cv_results["test-error-mean"]).tail(1))

#xgb_parameters = {'max_depth': [1,3,5], 'n_estimators': [2,5,10], 'learning_rate': [.01 , .1, .5]}
#print('XGB parameters are:')
#print(xgb_parameters)
##finding the best model
#xgb_optimal_model = grid_search(xgb.XGBClassifier(), xgb_parameters, X_train, Y_train)

#xgb.check_importance(xgb_model, X_train)









# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 19:05:14 2022

@author: Doug
"""

################
# LIBRARIES
################
import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import sys

import time
from timeit import default_timer as timer

import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import plot_tree
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib as mpl


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

#XGB Test vs Predict
unbalanced_weighting = len(Y_train.loc[Y_train == 1]) /len(Y_train.loc[Y_train == 0])

xgb_model = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.7, learning_rate = 0.8,
                max_depth = 9, alpha = 15, n_estimators = 10, tree_method='gpu_hist',num_parallel_tree=10,
                enable_categorical=True, scale_pos_weight = unbalanced_weighting)

xgb_model.fit(X_train,Y_train)

preds = xgb_model.predict(X_test)
preds_binary = np.round(preds)


print(accuracy_score(Y_test, preds_binary))
print(precision_score(Y_test,preds_binary))
print(recall_score(Y_test,preds_binary))



roc_auc_score(Y_test,preds_binary)

conf_matrix = confusion_matrix(y_true = Y_test, y_pred = preds_binary)
(conf_matrix[0,0]+conf_matrix[1,1])/np.sum(conf_matrix)

#mpl.use('Qt5Cairo')
#xgb_model.feature_importances_
#xgb.plot_importance(xgb_model)
#plt.rcParams['figure.figsize'] = [10, 50]
#plt.show()


#xgb.to_graphviz(xgb_model, num_trees=1)
#xgb.plot_tree(xgb_model,num_trees=0)

#pyplot.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
#pyplot.show()




#XGB Cross Val Zone --> Lower Alpha, Lower Depth = Less Conservative
params = {"objective":"reg:logistic",'colsample_bytree': 0.7,'learning_rate': 0.8,
                'max_depth': 8, 'alpha': 10,'num_parallel_tree':10}

df_dmatrix = xgb.DMatrix(data=X_train,label=Y_train, enable_categorical=True)

cv_results = xgb.cv(dtrain=df_dmatrix, params=params, nfold=5,
                    num_boost_round=50,early_stopping_rounds=10,metrics=['auc','error'], as_pandas=True, seed=123)

cv_results.head()
print((1-cv_results["test-error-mean"]).tail(1))











# FURTHER TESTING - IGNORE - PARAMETER TWEAKS AND BOOSTER CHANGES- NO USE
# xgb_model_gradient = xgb.XGBClassifier(objective ='reg:logistic', booster='dart',
#                 colsample_bytree = 0.3, learning_rate = 1,
#                 max_depth = 8, alpha = 10, n_estimators = 20,
#                 tree_method='gpu_hist',num_parallel_tree=50,
#                 subsample=0.1, colsample_bynode = 0.3, colsample_bylevel = 0.3, sampling_method="gradient_based",
#                 scale_pos_weight = unbalanced_weighting,
#                 enable_categorical=True)

# xgb_model_gradient.fit(X_train,Y_train)

# preds_gradient = xgb_model_gradient.predict(X_test)
# preds_gradient_binary = np.round(preds_gradient)


# accuracy_score(Y_test, preds_gradient_binary)
# precision_score(Y_test,preds_gradient_binary)
# recall_score(Y_test,preds_gradient_binary)
# roc_auc_score(Y_test,preds_gradient_binary)

# conf_matrix = confusion_matrix(y_true = Y_test, y_pred = preds_gradient_binary)
# (conf_matrix[0,0]+conf_matrix[1,1])/np.sum(conf_matrix)









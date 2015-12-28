"""

__author__ = 'amortized'
"""

from sklearn.cross_validation import train_test_split
import xgboost as xgb
import numpy as np;
from sklearn.cross_validation import train_test_split;
from sklearn.metrics import log_loss;
import pandas as pd;
from sklearn.grid_search import ParameterGrid;
from random import randint;
import sys;
import math
import copy

def generateParams():
    # Set the parameters by cross-validation
    paramaters_grid    = {'colsample_bytree' : [0.40],\
			  'subsample' : [0.40], 'max_depth' : [8]};


    paramaters_search  = list(ParameterGrid(paramaters_grid));

    parameters_to_try  = [];
    for ps in paramaters_search:
        params           = {'eta': 0.01, 'objective' : 'multi:softprob', 'nthread' : 32,\
			    'eval_metric' : 'mlogloss', 'seed' : randint(0,1000)};
        for param in ps.keys():
            params[str(param)] = ps[param];
        parameters_to_try.append(copy.copy(params));

    return parameters_to_try;     


def do(train_X, train_Y, param, num_round):
  X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.10, random_state=42);

  dtrain      = xgb.DMatrix( X_train, label=Y_train, missing=float('NaN'));
  dvalidation = xgb.DMatrix( X_test, label=Y_test,missing=float('NaN'));


  del X_train, X_test, Y_train, Y_test;

  #Track metrics on the watchlist
  watchlist = [ (dtrain,'train'), (dvalidation, 'validation') ];

  #Params
  param['num_class']  = len(np.unique(train_Y));

  classifier = xgb.train(param,dtrain,num_round,evals=watchlist,early_stopping_rounds=150);

  metric     = classifier.best_score;
  itr        = classifier.best_iteration;
  print("\n Metric : " + str(metric) + " for Params " + str(param) + " occurs at " + str(itr));

  del dtrain, dvalidation;

  return classifier, itr, metric;


train_X = np.load("/mnt/data/train_X.npy");
train_Y = np.load("/mnt/data/train_Y.npy");


parameters_to_try       = generateParams();
best_params             = None;
overall_best_metric     = sys.float_info.max;
overall_best_classifier = None;
overall_best_itr        = 0;

for i in range(0, len(parameters_to_try)):
  param     = parameters_to_try[i]
  num_round = 50000

  classifier, itr, metric = do(train_X, train_Y, param, num_round)

  if metric < overall_best_metric:
    overall_best_metric     = metric;
    best_params             = copy.copy(param);
    overall_best_itr        = itr; 
    overall_best_classifier = classifier;

  print("score : " + str(metric) + " for params : " +  str(param));

print("best score : " + str(overall_best_metric) + " for params : " +  str(best_params));

#Predict
test_X       = np.load("/mnt/data/test_X.npy");
test_visitNo = np.load("/mnt/data/test_visitNo.npy");

print("Test Data loaded");

test_X      = xgb.DMatrix(test_X, missing=float('NaN'));
y_hat       = overall_best_classifier.predict(test_X,ntree_limit=overall_best_itr)


df          = pd.read_csv("./data/sample_submission.csv");
columns     = df.columns;

test_df     = pd.DataFrame(columns=columns);

for i in range(0, len(test_visitNo)):
  test_df.loc[i] = [test_visitNo[i]] + list(y_hat[i]);

test_df.to_csv('./data/submission_d8.csv', index=False);


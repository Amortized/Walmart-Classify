"""

__author__ = 'amortized'
"""

from sklearn.cross_validation import train_test_split
import xgboost as xgb
import numpy as np;

#Read Data
def do(train_X, train_Y):
  X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.10, random_state=42);

  dtrain      = xgb.DMatrix( X_train, label=Y_train, missing=float('NaN'));
  dvalidation = xgb.DMatrix( X_test, label=Y_test,missing=float('NaN'));


  del X_train, X_test, Y_train, Y_test;

  #Track metrics on the watchlist
  watchlist = [ (dtrain,'train'), (dvalidation, 'validation') ];

  #Params
  param      = {'eval_metric' : 'mlogloss', 'objective' : 'multi:softprob', 'nthread' : 32, \
	            'colsample_bytree' : 1.0, 'subsample' : 1.0, 'eta': 0.02,\
	            'seed' : 42, 'num_class' : len(np.unique(train_Y)) };

  num_round  = 5000;

  classifier = xgb.train(param,dtrain,num_round,evals=watchlist,early_stopping_rounds=100);

  metric     = classifier.best_score;
  itr        = classifier.best_iteration;
  print("\n Metric : " + str(metric) + " for Params " + str(param) + " occurs at " + str(itr));

  del dtrain, dvalidation;

  return classifier, itr;


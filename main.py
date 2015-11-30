import pandas as pd;
import sys;
import numpy as np;
import warnings;
import math;
import model;
from sklearn import preprocessing;
import xgboost as xgb;
from multiprocessing import Pool;
from progressbar import ProgressBar, SimpleProgress;
import time;

def unique(column):
	'''
	 Generates a list of unique elements for a column
	'''
	tr   = np.unique(train[column]);
	tr   = [x for x in tr if str(x) != 'nan'];
	te   = np.unique(test[column]);
	te   = [x for x in te if str(x) != 'nan'];
	temp = list(set(tr) & set(te));
	return temp;

def UpcToKeep(train):
	upc   = train['Upc'].value_counts()
	upc_k = upc.keys();
	upc_v = upc.values;
	upc_v = upc_v / (1.0 * np.sum(upc_v));
	return set(np.asarray(upc_k)[upc_v > 0.0005]);


   
def genFeatures(visitNo, group, dpt, fln, upc, week, mode="train"):
	if mode == "train":
	  label   = np.unique(group['TripType'])[0];
	#Weekday feature
	weekday = np.unique(group['Weekday'])[0];
	weekend = 0;
	if weekday in ["Sunday", "Saturday"]:
	  weekend = 1;
	weekday_f = [0] * len(week);
	weekday_f[week.index(weekday)] = 1; 		
	#Scan count
	items_boughts = group[group['ScanCount'] > 0];
	items_sold    = group[group['ScanCount'] < 0];
	#Count of items bought/sold
	items_boughts_sum    = np.sum(items_boughts['ScanCount']);
	items_boughts_mean   = np.mean(items_boughts['ScanCount']);		
	items_boughts_median = np.median(items_boughts['ScanCount']);
	items_sold_sum       = np.sum(items_sold['ScanCount']);
	  
	#Unique 
	unique_upc_count     = 0;
	unique_dpt_count     = 0;
	unique_fln_count     = 0;
	#Department
	items_boughts_dpt_f  = [0] * (1 + len(dpt));
	if items_boughts.empty == False:
	  items_boughts_dpt    = items_boughts[[True if str(x) != 'nan' \
				   			 else False for x in items_boughts.DepartmentDescription]];
	  if items_boughts_dpt.empty == False:
	    unique_dpt_count = len(np.unique(items_boughts_dpt['DepartmentDescription']));
	    for name, group in items_boughts_dpt.groupby('DepartmentDescription'):
	   	  if name in dpt:
	  	    items_boughts_dpt_f[dpt.index(name)]  = np.sum(group['ScanCount']);
		  else:
		    items_boughts_dpt_f[len(items_boughts_dpt_f) - 1] = np.sum(group['ScanCount']);
		  
	#FinelineNumber
	items_boughts_fln_f  = [0] * (1 + len(fln));
	if items_boughts.empty == False:
	  items_boughts_fln    = items_boughts[[True if str(x) != 'nan' \
			                 else False for x in items_boughts.FinelineNumber]];
	  if items_boughts_fln.empty == False:
	    unique_fln_count = len(np.unique(items_boughts_fln['FinelineNumber']));    
	    for name, group in items_boughts_fln.groupby('FinelineNumber'):
		  if name in fln:
		    items_boughts_fln_f[fln.index(name)]  = np.sum(group['ScanCount']);
		  else:
		    items_boughts_fln_f[len(items_boughts_fln_f) - 1] = np.sum(group['ScanCount']);
	#Upc
	items_boughts_Upc_f  = [0] * (1 + len(upc));
	if items_boughts.empty == False:
	  items_boughts_Upc    = items_boughts[[True if str(x) != 'nan' \
			                 else False for x in items_boughts.Upc]];
	  if items_boughts_Upc.empty == False:
	    unique_upc_count = len(np.unique(items_boughts_Upc['Upc']));
	    for name, group in items_boughts_Upc.groupby('Upc'):
		  if name in upc:
		    items_boughts_Upc_f[upc.index(name)]  = np.sum(group['ScanCount']);        
		  else:
		    items_boughts_Upc_f[len(items_boughts_Upc_f) - 1] = np.sum(group['ScanCount']);
	  
	  
	features =  weekday_f + [weekend, items_boughts_sum, items_boughts_mean,\
		      items_boughts_median, items_sold_sum, unique_upc_count, \
		      unique_dpt_count, unique_fln_count]  + \
		      items_boughts_dpt_f + \
		      items_boughts_fln_f + \
		      items_boughts_Upc_f;
	              
	if mode == "train":            
	  #Feature
	  return (visitNo, features, label);
	else:
	  return (visitNo, features);

def gen_wrapper(args):
	return genFeatures(*args);

  

#Read the data 
train = pd.read_csv("./data/train.csv");
test  = pd.read_csv("./data/test.csv");
  
###############
dpt      = unique("DepartmentDescription");
fln      = unique("FinelineNumber");
upc      = list(set(unique("Upc")) & UpcToKeep(train));
week     = list(np.unique(train['Weekday']));
tripType = np.sort(np.unique(train.TripType));
le       = preprocessing.LabelEncoder();
le.fit(tripType);

#Prepare features at the visit level
train        = train.groupby('VisitNumber');

#Create a Thread pool.
pool         = Pool(3);

jobs         = [(visitNo, group, dpt, fln, upc, week, "train") \
				for visitNo,group in train]

results      = pool.imap_unordered(gen_wrapper, jobs);
pool.close() # No more work
while (True):
  completed = results._index
  if (completed == len(jobs)): break
  print "Waiting for", len(jobs)-completed, "tasks to complete..."
  time.sleep(2)



train_X      = [];
train_Y      = [];
for k in results:
  train_Y.append(k[2]);
  train_X.append(k[1]);



del train;  
train_X      = np.array(train_X);
train_Y      = np.array(train_Y);

#Ensure labels are between (0,num_class-1)
train_Y      = le.transform(train_Y);

#Train
best_model,best_n_trees = model.do(train_X, train_Y);

del train_X, train_Y;

#Test
test        = test.groupby('VisitNumber');

#Create a Thread pool.
pool         = Pool(3);
jobs         = [(visitNo, group, dpt, fln, upc, week, "test") \
				for visitNo,group in test]

results      = pool.imap_unordered(gen_wrapper, jobs);
pool.close() # No more work
while (True):
  completed = results._index
  if (completed == len(jobs)): break
  print "Waiting for", len(jobs)-completed, "tasks to complete..."
  time.sleep(2)


test_X       = [];
test_visitNo = [];
for k in results:
  test_X.append(k[1]);
  test_visitNo.append(k[0]);

del test;

test_X      = np.array(test_X);

test_X      = xgb.DMatrix(test_X, missing=float('NaN'));
y_hat       = best_model.predict(test_X,ntree_limit=best_n_trees)


df          = pd.read_csv("./data/sample_submission.csv");
columns     = df.columns;

#columns     = columns[:len(columns)-1]
test_df     = pd.DataFrame(columns=columns);

for i in range(0, len(test_visitNo)):
  test_df.loc[i] = [test_visitNo[i]] + list(y_hat[i]);

test_df.to_csv('./data/submission.csv', index=False);

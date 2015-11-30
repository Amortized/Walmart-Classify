import pandas as pd;
import sys;
import numpy as np;
import warnings;
import math;

#Read the data 
train = pd.read_csv("train.csv");
test  = pd.read_csv("test.csv");

def unique(column):
  '''
     Generates a list of unique elements for a column
  '''
  tr   = np.unique(train[column]);
  tr   = [x for x in tr if str(x) != 'nan'];
  te   = np.unique(test[column]);
  te   = [x for x in te if str(x) != 'nan'];
  temp = tr + te;
  temp = list(np.unique(np.asarray(temp)));
  return temp;

dpt = unique("DepartmentDescription");
fln = unique("FinelineNumber");
upc = unique("Upc");


#Prepare features at the visit level
train_visits = train.groupby('VisitNumber');

for name,group in train_visits:
  label   = np.unique(group['TripType'])[0];
  #Weekday feature
  weekday = np.unique(group['Weekday'])[0];
  #Scan count
  items_boughts = group[group['ScanCount'] > 0];
  items_sold    = group[group['ScanCount'] < 0];
  #Count of items bought/sold
  items_boughts_sum    = np.sum(items_boughts['ScanCount']);
  items_boughts_mean   = np.mean(items_boughts['ScanCount']);
  items_boughts_median = np.median(items_boughts['ScanCount']);
  items_sold_sum       = np.sum(items_sold['ScanCount']);
  #Department
  items_boughts_dpt_f  = [0] * len(dpt);
  if items_boughts.empty == False:
    items_boughts_dpt    = items_boughts[[True if str(x) != 'nan' else False for x in items_boughts.DepartmentDescription]];
    if items_boughts_dpt.empty == False:
      for name, group in items_boughts_dpt.groupby('DepartmentDescription'):
        items_boughts_dpt_f[dpt.index(name)]  = np.sum(group['ScanCount']);
  #FinelineNumber
  items_boughts_fln_f  = [0] * len(fln);
  if items_boughts.empty == False:
    items_boughts_fln    = items_boughts[[True if str(x) != 'nan' else False for x in items_boughts.FinelineNumber]];
    if items_boughts_fln.empty == False:
      for name, group in items_boughts_fln.groupby('FinelineNumber'):
        items_boughts_fln_f[fln.index(name)]  = np.sum(group['ScanCount']);
        




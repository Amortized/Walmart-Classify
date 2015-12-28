from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np;
from sklearn.cross_validation import train_test_split;
from sklearn.metrics import log_loss;
import pandas as pd;


train_X = np.load("./data/train_X.npy");
train_Y = np.load("./data/train_Y.npy");

print("Train Data loaded");

X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.10, random_state=42);


#RF Classifier
rfc = RandomForestClassifier(n_estimators=1000, max_depth=12, n_jobs = -1, min_samples_split = 50, min_samples_leaf = 50, max_features=1.0);

#Bagging
#bagging = BaggingClassifier(rfc, max_samples=0.8, max_features=0.8, n_jobs=6, verbose=1);

rfc.fit(X_train, Y_train);

Y_pred_test   = rfc.predict_proba(X_test);
Y_pred_train  = rfc.predict_proba(X_train);

test_score    = log_loss(Y_test, Y_pred_test);
train_score   = log_loss(Y_train, Y_pred_train);

print("Log loss on train set : " + str(train_score));
print("Log loss on validation set : " + str(test_score));

test_X       = np.load("./data/test_X.npy");
test_visitNo = np.load("./data/test_visitNo.npy");

print("Test Data loaded");

y_hat        = bagging.predict_proba(test_X);

df           = pd.read_csv("./data/sample_submission.csv");
columns      = df.columns;

test_df      = pd.DataFrame(columns=columns);

for i in range(0, len(test_visitNo)):
  test_df.loc[i] = [int(test_visitNo[i])] + list(y_hat[i]);

test_df.to_csv('./data/submission.csv', index=False);

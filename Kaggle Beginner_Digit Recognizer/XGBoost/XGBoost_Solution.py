# coding: utf-8

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import train_test_split
import os
import time

def get_features(train):
    output = list(train.columns)
    output.remove('label')
    return output

def findErrorValue(act, pred):
    if len(act) != len(pred):
        print('Length error!')
        exit()

    correct = 0
    for i in range(len(act)):
        if pred[i] == act[i]:
            correct += 1

    return correct/len(act)

print("Load train.csv")
train = pd.read_csv("../input/train.csv")
features = get_features(train)

print("Load test.csv")
test = pd.read_csv("../input/test.csv")

print('Features: ' + str(features))

random_state = 51
test_size = 0.1
params = {
    "objective": "multi:softmax",
    "num_class": 10,
    "booster" : "gbtree",
    # "gamma" : 30,
    "eval_metric": "merror",
    "eta": 0.01,
    "max_depth": 12,
    "silent": 1,
    "subsample": 0.95,
    "colsample_bytree": 0.9,
    "seed": random_state
}
num_boost_round = 10
early_stopping_rounds = 10

print("Split train")
X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
y_train = X_train['label']
y_valid = X_valid['label']

dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_ntree_limit)
correct = findErrorValue(y_valid.values, yhat)
print('Correct value: {:.6f}'.format(correct))

out = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_ntree_limit)
f = open("result_" + str(correct) + '_' + str(time.strftime("%Y-%m-%d-%H-%M")) + ".csv", 'w')
res_str = 'ImageId,Label\n'
total = 1
for i in range(len(out)):
    pstr = str(out[i].astype('int'))
    res_str += str(total) + ',' + pstr + '\n'
    total += 1
f.write(res_str)
import copy
import random
import shelve
import numpy as np
import xgboost as xgb
from scipy.sparse import load_npz, csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize, StandardScaler, MaxAbsScaler


def run_train_test():
    print('reading data...')
    with np.load('data/data.npz') as data:
        x_all, y_all = data['x'], data['y']

    print('transforming data...')
    y_all[y_all < 0] = 0

    # split train and test
    num_train = int(len(y_all) * 0.8)
    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_test, y_test = x_all[num_train:], y_all[num_train:]
    perm = np.random.permutation(num_train)
    x_train, y_train = x_train[perm], y_train[perm]

    # feature scaling
    scalar = MaxAbsScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)

    # to DMatrix
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    
    print('fitting xgb...')
    param = dict(objective='binary:logistic',
                 eval_metric=['error', 'auc'],
                 max_depth=7,
                 eta=0.3,
                 # scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
                 )
    print(param)
    bst = xgb.train(param, dtrain, num_boost_round=64, evals=[(dtrain, 'train'), (dtest, 'test')])
    bst.save_model('data/xgb.model')

    print('validating...')
    bst = xgb.Booster()
    bst.load_model('data/xgb.model')
    y_pred_proba = bst.predict(dtest)
    y_pred = y_pred_proba.copy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))


if __name__ == '__main__':
    run_train_test()

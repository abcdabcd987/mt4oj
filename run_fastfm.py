import copy
import random
import shelve
import numpy as np
import fastFM.sgd
from scipy.sparse import load_npz, csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize, StandardScaler, MaxAbsScaler


def run_train_test():
    print('reading data...')
    with np.load('data/data.npz') as data:
        x_all, y_all = data['x'], data['y']
    y_all = y_all.astype(np.int8)
    y_all[y_all == 0] = -1

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

    # sparse matrix
    x_train = csr_matrix(x_train)
    x_test = csr_matrix(x_test)
    
    print('fitting the fm...')
    fm = fastFM.sgd.FMClassification(n_iter=100000, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=0.001)
    fm.fit(x_train, y_train)

    print('validating...')
    y_pred = fm.predict(x_test)
    y_pred_proba = fm.predict_proba(x_test)
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))


if __name__ == '__main__':
    run_train_test()
    # run_recommend()

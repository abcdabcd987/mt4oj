import numpy as np
import fastFM.sgd
from scipy.sparse import load_npz, csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize


def main():
    print('reading data...')
    with np.load('data/data.npz') as data:
        x_all, y_all = data['x'], data['y']
    x_all = csr_matrix(normalize(x_all))
    num_train = int(len(y_all) * 0.8)
    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_test, y_test = x_all[num_train:], y_all[num_train:]
    
    print('fitting the fm...')
    fm = fastFM.sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=0.1)
    fm.fit(x_train, y_train)

    print('validating...')
    y_pred = fm.predict(x_test)
    y_pred_proba = fm.predict_proba(x_test)
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))


if __name__ == '__main__':
    main()

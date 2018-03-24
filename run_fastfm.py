from fastFM import sgd
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score


def read_libsvm_file(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]

def main():
    print('reading data...')
    x_train, y_train = read_libsvm_file('data/train.txt.gz')
    x_test, y_test = read_libsvm_file('data/test.txt.gz')
    
    print('fitting the fm...')
    fm = sgd.FMClassification(n_iter=1000,
                              init_stdev=0.1,
                              l2_reg_w=0,
                              l2_reg_V=0,
                              rank=2,
                              step_size=0.1)
    fm.fit(x_train, y_train)

    print('predicting...')
    y_pred = fm.predict(x_test)
    y_pred_proba = fm.predict_proba(x_test)
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))


if __name__ == '__main__':
    main()


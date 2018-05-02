import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import copy
import random
import shelve
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


def run_train_test():
    print('reading data...')
    with np.load('data/data.npz') as data:
        x_all, y_all = data['x'], data['y']

    print('transforming data...')
    y_all[y_all < 0] = 0

    # split train and test
    print('shuffling training data...')
    num_train = int(len(y_all) * 0.8)
    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_test, y_test = x_all[num_train:], y_all[num_train:]
    perm = np.random.permutation(num_train)
    x_train, y_train = x_train[perm], y_train[perm]

    # feature scaling
    print('scaling features...')
    scalar = MaxAbsScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)

    # reshape to 3D
    print('reshaping input matrix to 3D...')
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    # keras: build graph
    print('building graph...')
    model = Sequential()
    model.add(LSTM(100, input_shape=x_train.shape[1:]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # keras: train model
    print('training...')
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

    # validate
    print('validating...')
    y_pred_proba = model.predict(y_test)
    y_pred = y_pred_proba.copy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))


if __name__ == '__main__':
    run_train_test()


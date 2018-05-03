import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import bisect
import copy
import random
import shelve
import shutil
import numpy as np
from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.utils import Sequence
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler


class InputSequenceSharedData:
    def __init__(self, *, frac_train):
        print('InputSequenceSharedData: reading data..')
        with np.load('data/features.npz') as data:
            self.data_user_ids = data['user_id']
            self.num_rows = int(data['num_rows'])
            self.num_users = int(data['num_users'])
        with np.load('data/data.npz') as data:
            self.data_x, self.data_y = data['x'], data['y']

        print('InputSequenceSharedData: scaling features...')
        self.scalar = MaxAbsScaler().fit(self.data_x)
        self.data_x = self.scalar.transform(self.data_x)

        print('InputSequenceSharedData: building data structures...')
        self.num_train = int(self.num_rows * frac_train)
        self.train_user_rows = [[] for _ in range(self.num_users)]
        self.test_user_rows = [[] for _ in range(self.num_users)]
        for row in range(self.num_train):
            user_id = self.data_user_ids[row]
            self.train_user_rows[user_id].append(row)
        for row in range(self.num_train, self.num_rows):
            user_id = self.data_user_ids[row]
            self.test_user_rows[user_id].append(row)

        random_state = np.random.RandomState(1234)
        self.perm = random_state.permutation(self.num_users)
        self.train_user_offsets, self.test_user_offsets = [0], [0]
        for user_id in self.perm:
            self.train_user_offsets.append(self.train_user_offsets[-1] + len(self.train_user_rows[user_id]))
            self.test_user_offsets.append(self.test_user_offsets[-1] + len(self.test_user_rows[user_id]))
        self.train_user_offsets = self.train_user_offsets[1:]
        self.test_user_offsets = self.test_user_offsets[1:]

        print('InputSequenceSharedData: init done')


class InputSequence(Sequence):
    def __init__(self, data, *, is_train, batch_size, lookback):
        self._data = data
        self._is_train = is_train
        self._batch_size = batch_size
        self._lookback = lookback
        self._total_rows = self._data.num_train if self._is_train else self._data.num_rows - self._data.num_train

    def __len__(self):
        return int(np.ceil(self._total_rows / self._batch_size))

    def __getitem__(self, batch_idx):
        start_idx = batch_idx * self._batch_size
        end_idx = min((batch_idx+1) * self._batch_size, self._total_rows)
        ret_rows = end_idx - start_idx
        batch_x = np.zeros((ret_rows, self._lookback+1, self._data.data_x.shape[1]), dtype=np.float32)
        batch_y = np.empty(ret_rows, dtype=np.int8)
        user_offsets = self._data.train_user_offsets if self._is_train else self._data.test_user_offsets
        user_rows = self._data.train_user_rows if self._is_train else self._data.test_user_rows
        for i, idx in enumerate(range(start_idx, end_idx)):
            user_idx = bisect.bisect_right(user_offsets, idx)
            user_id = self._data.perm[user_idx]
            user_row_offset = idx - user_offsets[user_idx-1] if user_idx != 0 else idx
            for t in range(self._lookback+1):
                if user_row_offset-t >= 0:
                    row = user_rows[user_id][user_row_offset]
                    batch_x[i, t] = self._data.data_x[row]
                elif not self._is_train:
                    train_rows = self._data.train_user_rows[user_id]
                    row_offset = len(train_rows) + (user_row_offset - t)
                    if row_offset >= 0:
                        row = train_rows[row_offset]
                        batch_x[i, t] = self._data.data_x[row]
                # otherwise just fill 0
            row = user_rows[user_id][user_row_offset]
            batch_y[i] = self._data.data_y[row]
        return batch_x, batch_y


def run_train_test():
    batch_size = 256
    lookback = 5

    data = InputSequenceSharedData(frac_train=0.8)
    train_input_seq = InputSequence(data, is_train=True, batch_size=batch_size, lookback=lookback)
    test_input_seq = InputSequence(data, is_train=False, batch_size=batch_size, lookback=lookback)
    batch_x, batch_y = train_input_seq[20]
    x_test, y_test = [], []
    for i in range(len(test_input_seq)):
        batch_x, batch_y = test_input_seq[i]
        x_test.append(batch_x)
        y_test.append(batch_y)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    print(x_test.shape, y_test.shape)

    # keras: build graph
    print('building graph...')
    model = Sequential()
    model.add(LSTM(100, input_shape=(lookback+1, data.data_x.shape[1])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # keras: train model
    print('training...')
    for epoch in range(1, 13):
        model.fit_generator(train_input_seq,
            epochs=1, verbose=1,
            use_multiprocessing=True, workers=2,
            shuffle=False)
        model.save('data/rnn/model.h5')
        shutil.copy('data/rnn/model.h5', 'data/rnn/model-epoch-{}.h5'.format(epoch))

        # validate
        y_pred_proba = model.predict(x_test)
        y_pred = y_pred_proba.copy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        print('epoch', epoch, 'acc:', accuracy_score(y_test, y_pred))
        print('epoch', epoch, 'auc:', roc_auc_score(y_test, y_pred_proba))

    # validate
    print('validating...')
    model = load_model('data/rnn/model.h5')
    y_pred_proba = model.predict(x_test)
    y_pred = y_pred_proba.copy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))


if __name__ == '__main__':
    run_train_test()


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import bisect
import copy
import random
import shelve
import logging
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize, MaxAbsScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape, Flatten, LSTM
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)  # to allow a3c running more workers
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def np_divide(a, b):
    # see: https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


class Environment:
    def __init__(self, lookback):
        self._lookback = lookback

        logging.info('reading data')
        with shelve.open('data/problem_features.shelf', flag='r') as shelf:
            self._pf = shelf['pf']
        with np.load('data/features.npz') as data:
            data_user_id = data['user_id']
            self._data_y = data['label']
            self._data_problem_id = data['problem_id']
            self._data_num_rows = int(data['num_rows'])
            data_num_users = int(data['num_users'])
        with np.load('data/data.npz') as data:
            self._data_x = data['x']
        self._num_problems = len(self._pf)
        self._num_tags = len(next(iter(self._pf.values()))['pf_tags'])
        self._num_user_features = 2+2*self._num_tags
        self._sorted_problem_ids = sorted(self._pf.keys())
        self._map_idx_problem_id = dict(enumerate(self._sorted_problem_ids))

        logging.info('transforming data')
        self._scalar = MaxAbsScaler().fit(self._data_x)

        logging.info('loadding rnn model')
        self._rnn = load_model('data/rnn/model.h5')

        logging.info('collecting init states')
        self._user_rows = [[] for _ in range(data_num_users)]
        for row in range(self._data_num_rows):
            user_id = data_user_id[row]
            self._user_rows[user_id].append(row)
        self._user_offsets = [0]
        for user_id in range(data_num_users):
            self._user_offsets.append(self._user_offsets[-1] + len(self._user_rows[user_id]))
        self._user_offsets = self._user_offsets[1:]

        logging.info('environment init done')


    def _get_problem_features(self, problem_id):
        p = self._pf[problem_id]
        pf_basic_features = np.array([
            p['pf_num_submit'],
            p['pf_ac_rate'],
            p['pf_avg_lines'],
            p['pf_avg_bytes'],
            p['pf_avg_time'],
            p['pf_avg_mem'],
            p['pf_avg_score'],
        ], np.float32)
        pf_tags = p['pf_tags'].astype(np.float32)
        return np.concatenate((pf_basic_features, pf_tags))


    def _calc_next_user_features(self, last_x, problem_id, is_accepted):
        # extract user features from the vector
        u_num_submit = last_x[0]
        u_num_ac = last_x[0] * last_x[1]
        u_num_tag_submit = last_x[2 : 2+self._num_tags]
        u_num_tag_ac = u_num_tag_submit * last_x[2+self._num_tags : 2+self._num_tags*2]

        # update features
        p = self._pf[problem_id]
        if is_accepted:
            u_num_ac += 1
            u_num_tag_ac += p['pf_tags']
        u_num_submit += 1
        u_num_tag_submit += p['pf_tags']

        # form the vector
        uf_basic_features = np.array([
            u_num_submit,
            u_num_ac / u_num_submit if u_num_submit else 0,
        ], np.float32)
        uf_tag_ac_rate = np_divide(u_num_tag_ac, u_num_tag_submit)
        return np.concatenate((uf_basic_features, u_num_tag_submit, uf_tag_ac_rate))


    def _calc_student_score(self):
        x_lookback = np.zeros((self._lookback, self._data_x.shape[1]), dtype=np.float32)
        x_lookback[:, :self._num_user_features] = self._cur_user_features[1:]
        for t in range(self._lookback):
            problem_id = self._cur_problem_ids[t]
            if problem_id is not None:
                x_lookback[t, self._num_user_features:] = self._get_problem_features(problem_id)

        x = np.empty((self._num_problems, self._lookback+1, self._data_x.shape[1]), dtype=np.float32)
        for i, problem_id in enumerate(self._sorted_problem_ids):
            x[i, 0, :self._num_user_features] = self._cur_next_user_features
            x[i, 0, self._num_user_features:] = self._get_problem_features(problem_id)
            x[i, 1:] = x_lookback

        self._cur_prob = self._rnn.predict(x).squeeze()
        return np.average(self._cur_prob)


    @property
    def num_actions(self):
        return self._num_problems


    def new_episode(self):
        self._cur_user_features = np.zeros((self._lookback+1, self._num_user_features), dtype=np.float32)
        self._cur_problem_ids = [None] * (self._lookback+1)
        idx = random.randrange(self._data_num_rows)
        user_id = bisect.bisect_right(self._user_offsets, idx)
        user_row_offset = idx - self._user_offsets[user_id-1] if user_id != 0 else idx
        for t in range(self._lookback, -1, -1):
            if user_row_offset-t >= 0:
                row = self._user_rows[user_id][user_row_offset]
                self._cur_user_features[t] = self._data_x[row, :self._num_user_features]
                self._cur_problem_ids[t] = self._data_problem_id[row]
        is_accepted = bool(self._data_y[row])
        self._cur_next_user_features = self._calc_next_user_features(self._cur_user_features[0], self._cur_problem_ids[0], is_accepted)

        self._cur_score = self._calc_student_score()
        self._cur_num_recommend = 0
        return self._cur_user_features


    def step(self, action):
        EPISODE_NUM_RECOMMEND = 50
        if self._cur_num_recommend >= EPISODE_NUM_RECOMMEND:
            logging.warning('recommend more than %d problems', EPISODE_NUM_RECOMMEND)
        problem_id = self._map_idx_problem_id[action]

        # pretend the user to solve this problem
        prob = self._cur_prob[action]
        num_tries = np.random.geometric(prob)

        # update states
        for i in range(num_tries):
            is_accepted = i+1 == num_tries
            cur_user_features = np.empty((self._lookback+1, self._num_user_features), dtype=np.float32)
            cur_user_features[0] = self._cur_next_user_features
            cur_user_features[1:] = self._cur_user_features[:-1]
            cur_problem_ids = [problem_id] + self._cur_problem_ids[:-1]
            self._cur_user_features = cur_user_features
            self._cur_problem_ids = cur_problem_ids
            self._cur_next_user_features = self._calc_next_user_features(self._cur_user_features[0], self._cur_problem_ids[0], is_accepted)

        # calc reward
        last_score = self._cur_score
        self._cur_score = self._calc_student_score()
        reward = self._cur_score - last_score
        self._cur_num_recommend += 1
        done = self._cur_num_recommend >= EPISODE_NUM_RECOMMEND
        return self._cur_user_features, reward, done


def test_env():
    env = Environment()
    num_actions = env.num_actions
    for _ in range(2):
        state = env.new_episode()
        logging.info('new episode')
        done = False
        sum_reward = 0
        while not done:
            action = random.randrange(num_actions)
            state, reward, done = env.step(action)
            sum_reward += reward
            print('sum_reward', sum_reward)

logging.getLogger().setLevel(logging.DEBUG)


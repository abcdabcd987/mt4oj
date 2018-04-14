import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import copy
import random
import shelve
import logging
import numpy as np
import fastFM.sgd
from scipy.sparse import load_npz, csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D


def np_divide(a, b):
    # see: https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def get_db():
    db = psycopg2.connect('dbname=online_judge')
    return db


def get_cur(db, *, named=False):
    name = 'named_cursor_{}'.format(time.time())
    return db.cursor(name=name if named else None, cursor_factory=psycopg2.extras.DictCursor)


class Environment:
    def __init__(self):
        logging.info('reading data')
        with shelve.open('data/problem_features.shelf') as shelf:
            self._pf = shelf['pf']
        with np.load('data/data.npz') as data:
            x_train, y_train = data['x'], data['y']
        self._num_problems = len(self._pf)
        self._num_tags = len(next(iter(self._pf.values()))['pf_tags'])
        self._sorted_problem_ids = sorted(self._pf.keys())
        self._map_idx_problem_id = dict(enumerate(self._sorted_problem_ids))

        # fit fm
        logging.info('fitting fm')
        perm = np.random.permutation(len(y_train))
        x_train, y_train = x_train[perm], y_train[perm]
        self._x_train = x_train
        self._fm = fastFM.sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=0.1)
        self._fm.fit(csr_matrix(normalize(x_train)), y_train)

        # collect init states
        logging.info('collecting init states')
        INIT_STATE_LEAST_NUM_AC = 12  # minimal number of ac for this record to become the RL init state
        init_state_idx_pool = np.empty(x_train.shape[0], np.float32)
        cnt_init = 0
        for i in range(x_train.shape[0]):
            num_ac = x_train[i, 0] * x_train[i, 1]
            if num_ac > INIT_STATE_LEAST_NUM_AC:
                init_state_idx_pool[cnt_init] = i
                cnt_init += 1
        self._init_state_idx_pool = init_state_idx_pool[:cnt_init]

        logging.info('environment init done')


    def _user_model_features_to_state(self, x):
        return x[:2+self._num_tags*2]


    def _state_to_user_model_features(self, s, p):
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
        return np.concatenate((s, pf_basic_features, pf_tags))


    def _calc_student_score(self):
        x = np.empty((self._num_problems, 9+self._num_tags*3), np.float32)
        for i, problem_id in enumerate(self._sorted_problem_ids):
            p = self._pf[problem_id]
            x[i] = self._state_to_user_model_features(self._cur_state, p)
        self._cur_prob = self._fm.predict_proba(csr_matrix(normalize(x)))
        return np.average(self._cur_prob)


    @property
    def num_actions(self):
        return self._num_problems


    def new_episode(self):
        state_idx = random.randrange(self._init_state_idx_pool.shape[0])
        self._cur_state = self._user_model_features_to_state(self._x_train[state_idx])
        self._cur_score = self._calc_student_score()
        self._cur_num_recommend = 0
        return self._cur_state


    def step(self, action):
        EPISODE_NUM_RECOMMEND = 10
        if self._cur_num_recommend >= EPISODE_NUM_RECOMMEND:
            logging.warning('recommend more than %d problems', EPISODE_NUM_RECOMMEND)
        problem_id = self._map_idx_problem_id[action]
        p = self._pf[problem_id]

        # extract user feature from the state
        u_num_submit = self._cur_state[0]
        u_num_ac = self._cur_state[0] * self._cur_state[1]
        u_num_tag_submit = self._cur_state[2 : 2+self._num_tags]
        u_num_tag_ac = u_num_tag_submit * self._cur_state[2+self._num_tags : 2+self._num_tags*2]

        # pretend the user to solve this problem
        prob = self._cur_prob[action]
        num_tries = np.random.geometric(prob)
        u_num_submit += num_tries
        u_num_ac += 1
        u_num_tag_submit += num_tries * p['pf_tags']
        u_num_tag_ac += p['pf_tags']

        # construct the state
        uf_ac_rate = u_num_ac / u_num_submit
        uf_tag_ac_rate = np_divide(u_num_tag_ac, u_num_tag_submit)
        uf_basic_features = np.array([u_num_submit, uf_ac_rate], np.float32)
        self._cur_state = np.concatenate((uf_basic_features, u_num_tag_submit, uf_tag_ac_rate))

        # calc reward
        last_score = self._cur_score
        self._calc_student_score()
        reward = self._cur_score - last_score
        self._cur_num_recommend += 1
        done = self._cur_num_recommend >= EPISODE_NUM_RECOMMEND
        return self._cur_state, reward, done


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


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_env()

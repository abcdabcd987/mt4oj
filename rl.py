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
        self._lookback = 5

        logging.info('reading data')
        with shelve.open('data/problem_features.shelf') as shelf:
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
        x_lookback = np.empty((self._lookback, self._data_x.shape[1]), dtype=np.float32)
        x_lookback[:, :self._num_user_features] = self._cur_user_features[1:]
        for t in range(self._lookback):
            problem_id = self._cur_problem_ids[t]
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



class PGAgent:
    # ref: https://github.com/keon/policy-gradient/blob/master/pg.py
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_shape=(self.state_size,)))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size], np.float32)
        y[action] = 1
        self.gradients.append(np.array(y) - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        # self.model.fit(X, Y, batch_size=1, callbacks=[self.cb_tensorboard])
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



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


def run_rl():
    env = Environment()
    state = env.new_episode()
    score = 0
    episode = 0

    state_size = len(state)
    action_size = env.num_actions
    agent = PGAgent(state_size, action_size)
    MODEL_FILENAME = 'data/pgagent.h5'
    if os.path.exists(MODEL_FILENAME):
        agent.load(MODEL_FILENAME)
    while True:
        # x = normalize(state)
        x = state
        action, prob = agent.act(x)
        state, reward, done = env.step(action)
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            agent.train()
            logging.info('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.new_episode()
            if episode > 1 and episode % 32 == 0:
                agent.save(MODEL_FILENAME)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_env()
    # run_rl()

from rl_common import *


class PGAgent:
    # ref: https://github.com/keon/policy-gradient/blob/master/pg.py
    def __init__(self, lookback, state_size, action_size):
        self.lookback = lookback
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
        model.add(Flatten(input_shape=(self.lookback+1, self.state_size)))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
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
        state = state.reshape([1] + list(state.shape))
        aprob = self.model.predict(state).flatten()
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


def run_rl():
    lookback = 5
    env = Environment(lookback)
    state = env.new_episode()
    score = 0
    episode = 0

    state_size = state.shape[1]
    action_size = env.num_actions
    agent = PGAgent(lookback, state_size, action_size)
    MODEL_FILENAME = 'data/pgagent.h5'
    # if os.path.exists(MODEL_FILENAME):
    #     agent.load(MODEL_FILENAME)
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
    run_rl()

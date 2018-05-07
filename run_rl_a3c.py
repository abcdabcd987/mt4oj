# ref: https://github.com/Grzego/async-rl/blob/master/a3c/train.py

from multiprocessing import *
from collections import deque
import numpy as np
import argparse
import traceback

# -----
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--lookback', default=5, type=int)
parser.add_argument('--processes', default=4, help='Number of processes that generate experience for agent', dest='processes', type=int)
parser.add_argument('--lr', default=0.001, help='Learning rate', dest='learning_rate', type=float)
parser.add_argument('--steps', default=80000000, help='Number of frames to decay learning rate', dest='steps', type=int)
parser.add_argument('--batch_size', default=32, help='Batch size to use during training', dest='batch_size', type=int)
parser.add_argument('--swap_freq', default=10, help='Number of frames before swapping network weights', dest='swap_freq', type=int)
parser.add_argument('--checkpoint', default=0, help='Frame to resume training', dest='checkpoint', type=int)
parser.add_argument('--save_freq', default=1000, help='Number of frames before saving weights', dest='save_freq', type=int)
parser.add_argument('--queue_size', default=256, help='Size of queue holding agent experience', dest='queue_size', type=int)
parser.add_argument('--beta', default=0.01, dest='beta', type=float)
# -----
args = parser.parse_args()


# -----


def build_network(input_shape, output_shape):
    from keras.models import Model
    from keras.layers import Input, Flatten, Dense, LSTM
    # -----
    state = Input(shape=input_shape)
    h = LSTM(10)(state)
    h = Dense(32, activation='relu')(h)

    value = Dense(1, activation='linear', name='value')(h)
    policy = Dense(output_shape, activation='softmax', name='policy')(h)

    value_network = Model(inputs=state, outputs=value)
    policy_network = Model(inputs=state, outputs=policy)

    reward = Input(shape=(1,))
    advantage = reward - value
    train_network = Model(inputs=[state, reward], outputs=[value, policy])

    return value_network, policy_network, train_network, advantage, reward


def policy_loss(advantage, beta=0.01):
    from keras import backend as K

    def loss(y_true, y_pred):
        return -K.sum(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()) * K.flatten(advantage)) + \
               beta * K.sum(y_pred * K.log(y_pred + K.epsilon()))

    return loss


def value_loss():
    from keras import backend as K

    def loss(y_true, y_pred):
        return 0.5 * K.sum(K.square(y_true - y_pred))

    return loss


# -----

class LearningAgent(object):
    def __init__(self, lookback, state_size, action_size, batch_size, swap_freq):
        from keras.optimizers import RMSprop        
        # -----
        self.lookback = lookback
        self.state_size = state_size
        self.action_size = action_size
        self.observation_shape = (self.lookback+1, self.state_size)
        self.batch_size = batch_size

        value_net, policy_net, train_net, advantage, reward = build_network(self.observation_shape, self.action_size)
        self.train_net = train_net
        self.train_net.compile(optimizer=RMSprop(epsilon=0.1, rho=0.99),
                               loss=[value_loss(), policy_loss(advantage, args.beta)])

        self.swap_freq = swap_freq
        self.swap_counter = self.swap_freq
        self.unroll = np.arange(self.batch_size)
        self.targets = np.zeros((self.batch_size, self.action_size))
        self.counter = 0

    def learn(self, last_observations, actions, rewards, learning_rate=0.001):
        import keras.backend as K
        K.set_value(self.train_net.optimizer.lr, learning_rate)
        self.counter += 1
        # -----
        self.targets.fill(0.)
        self.targets[self.unroll, actions] = 1.
        # -----
        loss = self.train_net.train_on_batch([last_observations, rewards], [rewards, self.targets])
        self.swap_counter -= 1
        if self.swap_counter < 0:
            self.swap_counter += self.swap_freq
            return True
        return False


def learn_proc(mem_queue, weight_dict):
    import os
    from rl_common import Environment
    pid = os.getpid()
    # -----
    print(' %5d> Learning process' % (pid,))
    # -----
    save_freq = args.save_freq
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    steps = args.steps
    # -----
    env = Environment(args.lookback)
    state = env.new_episode()
    state_size = state.shape[1]
    action_size = env.num_actions
    del state
    del env
    agent = LearningAgent(args.lookback, state_size, action_size, batch_size=args.batch_size, swap_freq=args.swap_freq)
    # -----
    if checkpoint > 0:
        print(' %5d> Loading weights from file' % (pid,))
        agent.train_net.load_weights('data/a3c/%d.h5' % (checkpoint,))
        # -----
    print(' %5d> Setting weights in dict' % (pid,))
    weight_dict['update'] = 0
    weight_dict['weights'] = agent.train_net.get_weights()
    # -----
    last_obs = np.zeros((batch_size,) + agent.observation_shape)
    actions = np.zeros(batch_size, dtype=np.int32)
    rewards = np.zeros(batch_size)
    # -----
    idx = 0
    agent.counter = checkpoint
    save_counter = checkpoint % save_freq + save_freq
    while True:
        # -----
        last_obs[idx, ...], actions[idx], rewards[idx] = mem_queue.get()
        idx = (idx + 1) % batch_size
        if idx == 0:
            lr = max(0.00000001, (steps - agent.counter) / steps * learning_rate)
            updated = agent.learn(last_obs, actions, rewards, learning_rate=lr)
            if updated:
                # print(' %5d> Updating weights in dict' % (pid,))
                weight_dict['weights'] = agent.train_net.get_weights()
                weight_dict['update'] += 1
        # -----
        save_counter -= 1
        if save_counter < 0:
            save_counter += save_freq
            agent.train_net.save_weights('data/a3c/%d.h5' % (agent.counter,), overwrite=True)


class ActingAgent(object):
    def __init__(self, lookback, state_size, action_size, discount=0.99):
        self.lookback = lookback
        self.state_size = state_size
        self.action_size = action_size
        self.observation_shape = (self.lookback+1, self.state_size)

        value_net, policy_net, train_net, advantage, reward = build_network(self.observation_shape, self.action_size)
        self.policy_net = policy_net
        self.load_net = train_net

        self.observations = np.zeros(self.observation_shape)
        self.last_observations = np.zeros_like(self.observations)
        # -----
        self.episode_observations = deque()
        self.episode_actions = deque()
        self.episode_rewards = deque()
        self.discount = discount
        self.counter = 0

    def init_episode(self, observation):
        for _ in range(self.lookback+1):
            self.save_observation(observation)

    def reset(self):
        self.counter = 0
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def sars_data(self, action, reward, observation, terminal, mem_queue):
        self.save_observation(observation)
        # -----
        self.episode_observations.appendleft(self.last_observations)
        self.episode_actions.appendleft(action)
        self.episode_rewards.appendleft(reward)
        # -----
        self.counter += 1
        if terminal:
            r = 0.
            for i in range(self.counter):
                r = self.episode_rewards[i] + self.discount * r
                mem_queue.put((self.episode_observations[i], self.episode_actions[i], r))
            self.reset()

    def choose_action(self):
        policy = self.policy_net.predict(self.observations[None, ...])[0]
        return np.random.choice(np.arange(self.action_size), p=policy)

    def save_observation(self, observation):
        self.last_observations = self.observations
        self.observations = observation


def generate_experience_proc(mem_queue, weight_dict, no, episode_reward_queue):
    import os
    from rl_common import Environment
    pid = os.getpid()
    # -----
    print(' %5d> Process started' % (pid,))
    # -----
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    # -----
    env = Environment(args.lookback)
    state = env.new_episode()
    state_size = state.shape[1]
    action_size = env.num_actions

    agent = ActingAgent(args.lookback, state_size, action_size)

    if checkpoint > 0:
        agent.load_net.load_weights('data/a3c/%d.h5' % (checkpoint,))
        print(' %5d> Loaded weights from file' % (pid,))
    else:
        import time
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        agent.load_net.set_weights(weight_dict['weights'])
        print(' %5d> Loaded weights from dict' % (pid,))

    last_update = 0
    while True:
        done = False
        episode_reward = 0
        observation = env.new_episode()
        agent.init_episode(observation)

        # -----
        while not done:
            frames += 1
            action = agent.choose_action()
            observation, reward, done = env.step(action)
            episode_reward += reward
            # -----
            agent.sars_data(action, reward, observation, done, mem_queue)
            # -----
            if frames % batch_size == 0:
                update = weight_dict.get('update', 0)
                if update > last_update:
                    last_update = update
                    # print(' %5d> Getting weights from dict' % (pid,))
                    agent.load_net.set_weights(weight_dict['weights'])
        # -----
        episode_reward_queue.put(episode_reward)


def init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def error(msg, *args):
    return get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result


def main():
    manager = Manager()
    weight_dict = manager.dict()
    mem_queue = manager.Queue(args.queue_size)
    episode_reward_queue = manager.Queue()
    pool = Pool(args.processes + 1, init_worker)
    try:
        for i in range(args.processes):
            pool.apply_async(LogExceptions(generate_experience_proc), (mem_queue, weight_dict, i, episode_reward_queue))
        pool.apply_async(LogExceptions(learn_proc), (mem_queue, weight_dict))
        pool.close()

        latest_rewards = deque(maxlen=10)
        cnt = 0
        ema = 0
        while True:
            reward = episode_reward_queue.get()
            latest_rewards.appendleft(reward)
            ema = ema * 0.99 + reward * 0.01
            cnt += 1
            recent = ', '.join(map(lambda x: '{:+.4f}'.format(x), latest_rewards))
            print('\repisode {} reward ema {:+.6f} recent [{}]'.format(cnt, ema, recent), end='')
            if cnt % 50 == 0:
                print('')

        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()
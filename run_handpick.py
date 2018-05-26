from rl_common import *
from rl_a3c import ActingAgent

class FakeQueue:
    def put(self, x):
        pass

def main():
    np.random.seed(123)
    with shelve.open('data/handpick.shelf') as shelf:
        problems = shelf['problems']
        users = shelf['users']

    lookback = 5
    env = Environment(lookback=lookback, episode_length=len(problems))
    state = env.new_episode()
    state_size = state.shape[1]
    action_size = env.num_actions

    mem_queue = FakeQueue()
    agent = ActingAgent(lookback, state_size, action_size, n_step=5)
    agent.load_net.load_weights('data/a3c/a3c.h5')

    f = open('data/handpick.csv', 'w')
    f.write('\t'.join(['user_id', 'init_score', 'last_score', 'greedy_score', 'random_score', 'rl_score']) + '\n')
    for user in users:
        # handpick
        env.new_episode(row_idx=user['last'])
        last_score = env._cur_score
        env.new_episode(row_idx=user['init'])
        init_score = env._cur_score
        print('init:{:.3f} last:{:.3f}\r'.format(init_score, last_score), end='')

        # greedy
        env.new_episode(row_idx=user['init'])
        done = False
        while not done:
            action = np.argmax(env._cur_prob)
            observation, reward, done = env.step(action)
        greedy_score = env._cur_score
        print('init:{:.3f} last:{:.3f} greedy:{:.3f}\r'.format(init_score, last_score, greedy_score), end='')

        # random
        random_score = 0
        for cnt_try in range(50):
            env.new_episode(row_idx=user['init'])
            done = False
            while not done:
                action = np.argmax(env._cur_prob)
                observation, reward, done = env.step(action)
            random_score = max(random_score, env._cur_score)
            print('init:{:.3f} last:{:.3f} greedy:{:.3f} | try:{:d} current:{:.3f} best:{:.3f}\r'.format(init_score, last_score, greedy_score, cnt_try+1, env._cur_score, random_score), end='')

        # rl
        rl_score = 0
        for cnt_try in range(50):
            observation = env.new_episode(row_idx=user['init'])
            agent.init_episode(observation)
            done = False
            while not done:
                action = agent.choose_action()
                observation, reward, done = env.step(action)
                agent.sars_data(action, reward, observation, done, mem_queue)
            rl_score = max(rl_score, env._cur_score)
            print('init:{:.3f} last:{:.3f} greedy:{:.3f} random:{.3f} | try:{:d} current:{:.3f} best:{:.3f}\r'.format(init_score, last_score, greedy_score, random_score, cnt_try, env._cur_score, rl_score), end='')
        
        print('')
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(user['user_id'], init_score, last_score, greedy_score, random_score, rl_score))
        f.flush()
    f.close()


if __name__ == '__main__':
    main()

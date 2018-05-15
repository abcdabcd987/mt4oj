import argparse
import multiprocessing
import random
import time
import os
from datetime import datetime

parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--processes', default=4, type=int)
parser.add_argument('--episodes', default=1000000, type=int)
args = parser.parse_args()


def worker(queue):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from rl_common import Environment
    env = Environment(lookback=5)
    num_actions = env.num_actions

    while True:
        state = env.new_episode()
        done = False
        sum_reward = 0
        while not done:
            action = random.randrange(num_actions)
            state, reward, done = env.step(action)
            sum_reward += reward
        queue.put(sum_reward)


def main():
    today = datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = os.path.join('logs', 'random-{}.log'.format(today))
    logf = open(filename, 'w')

    queue = multiprocessing.Queue()
    workers = []
    for _ in range(args.processes):
        p = multiprocessing.Process(target=worker, args=(queue,))
        p.start()
        workers.append(p)


    try:
        cnt = 0
        start_time = time.time()
        newline_time = start_time
        for _ in range(args.episodes):
            reward = queue.get()

            cnt += 1
            now = time.time()
            elapse = int(now - start_time)
            h = elapse // 3600
            m = elapse // 60 % 60
            s = elapse % 60
            episode_per_minute = cnt / elapse * 60
            print('\r[{:02d}h{:02d}m{:02d}s]episode:{} ({:.0f} episodes/m)'.format(h, m, s, cnt, episode_per_minute), end='')
            if now-newline_time > 60.:
                newline_time = now
                print('')
            logf.write('{:.3f}\t{:+.16f}\n'.format(now - start_time, reward))
            logf.flush()
    except KeyboardInterrupt:
        pass
    logf.close()
    for p in workers:
        p.terminate()
    for p in workers:
        p.join()


if __name__ == '__main__':
    main()

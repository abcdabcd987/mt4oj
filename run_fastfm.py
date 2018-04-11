import copy
import random
import shelve
import numpy as np
import fastFM.sgd
from scipy.sparse import load_npz, csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize


def calc_features(u, p):
    return [
        u['num_submit'],
        u['num_ac'] / u['num_submit'] if u['num_submit'] else 0,
        p['pf_num_submit'],
        p['pf_ac_rate'],
        p['pf_avg_lines'],
        p['pf_avg_bytes'],
        p['pf_avg_time'],
        p['pf_avg_mem'],
        p['pf_avg_score'],
    ]


def run_train_test():
    print('reading data...')
    with np.load('data/data.npz') as data:
        x_all, y_all = data['x'], data['y']
    x_all = csr_matrix(normalize(x_all))
    num_train = int(len(y_all) * 0.8)
    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_test, y_test = x_all[num_train:], y_all[num_train:]
    
    print('fitting the fm...')
    fm = fastFM.sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=0.1)
    fm.fit(x_train, y_train)

    print('validating...')
    y_pred = fm.predict(x_test)
    y_pred_proba = fm.predict_proba(x_test)
    print('acc:', accuracy_score(y_test, y_pred))
    print('auc:', roc_auc_score(y_test, y_pred_proba))


def run_recommend():
    print('read data')
    with shelve.open('data/recommend.shelf') as shelf:
        x_train = shelf['x_train']
        y_train = shelf['y_train']
        user_list = shelf['user_list']
        user_features = shelf['user_features']
        report_problem_list = shelf['report_problem_list']
    with shelve.open('data/problem_features.shelf') as shelf:
        pf = shelf['pf']
    problem_list = sorted(pf.keys())
    csv = {u: {} for u in user_list}

    def gen_eval_data(users):
        cnt_eval = len(problem_list) * len(user_list)
        x_eval = np.empty((cnt_eval, 9), dtype=np.float32)
        i = 0
        for student_id in user_list:
            u = users[student_id]
            for problem_id in problem_list:
                p = pf[problem_id]
                features = calc_features(u, p)
                for j, v in enumerate(features):
                    x_eval[i, j] = v
                i += 1
        return csr_matrix(normalize(x_eval))
    

    def write_ability(x_eval, prob, column_name):
        for i, student_id in enumerate(user_list):
            csv[student_id][column_name] = np.average(prob[i*len(problem_list) : (i+1)*len(problem_list)])


    # train
    print('train')
    x_train = csr_matrix(normalize(x_train))
    fm = fastFM.sgd.FMClassification(n_iter=1000, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=0.1)
    fm.fit(x_train, y_train)

    # calc ability_before
    print('before')
    x_eval = gen_eval_data(user_features)
    prob_before = fm.predict_proba(x_eval)
    write_ability(x_eval, prob_before, 'before')

    # calc ability_adhoc
    print('adhoc')
    users = copy.deepcopy(user_features)
    for i, student_id in enumerate(user_list):
        u = users[student_id]
        for problem_id in report_problem_list:
            index = problem_list.index(problem_id)
            p = prob_before[i * len(problem_list) + index]
            while True:
                u['num_submit'] += 1
                if random.random() < p:
                    u['num_ac'] += 1
                    break
    x_eval = gen_eval_data(users)
    prob_adhoc = fm.predict_proba(x_eval)
    write_ability(x_eval, prob_adhoc, 'adhoc')

    # calc ability_recommend
    print('recommend')
    users = copy.deepcopy(user_features)
    for i, student_id in enumerate(user_list):
        prob = prob_before[i*len(problem_list) : (i+1)*len(problem_list)]
        argsort = np.argsort(prob)[::-1]
        # recommend the top n problem
        u = users[student_id]
        for index in argsort[:len(report_problem_list)]:
            p = prob[index]
            while True:
                u['num_submit'] += 1
                if random.random() < p:
                    u['num_ac'] += 1
                    break
    x_eval = gen_eval_data(users)
    prob_recommend = fm.predict_proba(x_eval)
    write_ability(x_eval, prob_recommend, 'recommend')

    # write csv
    with open('data/recommend.csv', 'w') as f:
        f.write('user,before,adhoc,recommend\n')
        for student_id in user_list:
            d = csv[student_id]
            f.write('{0},{1[before]},{1[adhoc]},{1[recommend]}\n'.format(student_id, d))


if __name__ == '__main__':
    run_train_test()
    # run_recommend()

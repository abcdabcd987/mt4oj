import copy
import random
import shelve
import numpy as np
import fastFM.sgd
from scipy.sparse import load_npz, csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize, StandardScaler, MaxAbsScaler


def np_divide(a, b):
    # see: https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def calc_features(u, p):
    uf_basic_features = np.array([
        u['num_submit'],
        u['num_ac'] / u['num_submit'] if u['num_submit'] else 0,
    ], np.float32)
    uf_num_tag_ac = u['num_tag_ac'].astype(np.float32)
    uf_num_tag_submit = u['num_tag_submit'].astype(np.float32)
    uf_tag_ac_rate = np_divide(uf_num_tag_ac, uf_num_tag_submit)

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
    return np.concatenate((
        uf_basic_features,
        uf_num_tag_submit,
        uf_tag_ac_rate,
        pf_basic_features,
        pf_tags,
    ))


def run_train_test():
    print('reading data...')
    with np.load('data/data.npz') as data:
        x_all, y_all = data['x'], data['y']
    y_all = y_all.astype(np.int8)
    y_all[y_all == 0] = -1

    # split train and test
    num_train = int(len(y_all) * 0.8)
    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_test, y_test = x_all[num_train:], y_all[num_train:]
    perm = np.random.permutation(num_train)
    x_train, y_train = x_train[perm], y_train[perm]

    # feature scaling
    scalar = MaxAbsScaler().fit(x_train)
    x_train = scalar.transform(x_train)
    x_test = scalar.transform(x_test)

    # sparse matrix
    x_train = csr_matrix(x_train)
    x_test = csr_matrix(x_test)
    
    print('fitting the fm...')
    fm = fastFM.sgd.FMClassification(n_iter=100000, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=0.001)
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
    num_tags = len(next(iter(pf.values()))['pf_tags'])
    num_features = 9 + num_tags*3
    csv = {u: {} for u in user_list}

    def gen_eval_data(users):
        cnt_eval = len(problem_list) * len(user_list)
        x_eval = np.empty((cnt_eval, num_features), dtype=np.float32)
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
            p = pf[problem_id]
            index = problem_list.index(problem_id)
            chance = prob_before[i * len(problem_list) + index]
            while True:
                u['num_submit'] += 1
                u['num_tag_submit'] += p['pf_tags']
                if random.random() < chance:
                    break
            u['num_ac'] += 1
            u['num_tag_ac'] += p['pf_tags']
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
            p = pf[problem_id]
            chance = prob[index]
            while True:
                u['num_submit'] += 1
                u['num_tag_submit'] += p['pf_tags']
                if random.random() < chance:
                    break
            u['num_ac'] += 1
            u['num_tag_ac'] += p['pf_tags']
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

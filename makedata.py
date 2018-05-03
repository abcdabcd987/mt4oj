import re
import math
import yaml
import gzip
import time
import shelve
import psycopg2
import psycopg2.extras
import multiprocessing
import numpy as np
from tqdm import tqdm


import fastFM.sgd
from scipy.sparse import load_npz, csr_matrix, save_npz
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import normalize


def np_divide(a, b):
    # see: https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def np_log(x):
    return np.log(x) if x != 0 else 0


def get_db():
    db = psycopg2.connect('dbname=online_judge')
    return db


def get_cur(db, *, named=False):
    name = 'named_cursor_{}'.format(time.time())
    return db.cursor(name=name if named else None, cursor_factory=psycopg2.extras.DictCursor)


def parse_judge_message(msg):
    # return (verdict, time in ms, memory in kb)
    pattern = r'(.+?) \(Time: (\d+)ms, Memory: (\d+)kb\)'
    res = re.findall(pattern, msg)
    return list(map(lambda x: (x[0], int(x[1]), int(x[2])), res))


def get_problem_tags():
    with open('data/problem_tags.yaml') as f:
        problem_tags_yaml = yaml.load(f)
    all_tags = set(tag for tag_list in problem_tags_yaml.values() for tag in tag_list)
    map_tag_idx = {tag: i for i, tag in enumerate(all_tags)}
    problem_tags = {}
    for problem_id, tag_list in problem_tags_yaml.items():
        onehot = np.zeros(len(all_tags), np.int8)
        for tag in tag_list:
            idx = map_tag_idx[tag]
            onehot[idx] = 1
        problem_tags[problem_id] = onehot
    return problem_tags


def pool_initializer():
    global db
    db = get_db()


def extract_single_problem_features(problem_id):
    global db
    cur = get_cur(db, named=True)

    num_records = 0
    num_accepted = 0
    avg_code_lines = 0
    avg_code_bytes = 0
    avg_run_time = 0
    avg_run_mem = 0
    avg_score = 0
    cur.execute('SELECT * FROM records WHERE problem_id=%s AND language=%s', (problem_id, 'C++'))
    for record in cur:
        if record['result'] in ['Compile Error', 'System Error', 'Unknown']:
            continue

        num_records += 1
        testcases = parse_judge_message(record['judge_message'])
        avg_score += sum(map(lambda x: x[0] == 'Accepted', testcases)) / len(testcases)

        if record['result'] != 'Accepted':
            continue
        num_accepted += 1
        avg_code_lines += len(record['submit_code'].splitlines())
        avg_code_bytes += len(record['submit_code'])
        avg_run_time += sum(map(lambda x: x[1], testcases))
        avg_run_mem += max(map(lambda x: x[2], testcases))

    if num_accepted:
        avg_code_lines /= num_accepted
        avg_code_bytes /= num_accepted
        avg_run_time /= num_accepted
        avg_run_mem /= num_accepted
        avg_score /= num_records
    
    f = dict(pf_num_submit=num_records,
             pf_ac_rate=num_accepted / num_records if num_records else 0,
             pf_avg_lines=avg_code_lines,
             pf_avg_bytes=avg_code_bytes,
             pf_avg_time=avg_run_time,
             pf_avg_mem=avg_run_mem,
             pf_avg_score=avg_score)
    cur.close()
    return problem_id, f


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


def run_make_problem_features(db):
    cur = get_cur(db)
    cur.execute('SELECT id FROM problems')
    problem_ids = [row['id'] for row in cur]
    pf = {}
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=pool_initializer)
    for problem_id, f in tqdm(pool.imap_unordered(extract_single_problem_features, problem_ids), total=len(problem_ids)):
        pf[problem_id] = f
    pool.close()
    pool.join()

    problem_tags = get_problem_tags()
    for problem_id, tags in problem_tags.items():
        pf[problem_id]['pf_tags'] = tags

    with shelve.open('data/problem_features.shelf') as db:
        db['pf'] = pf
    cur.close()


def run_makedata(db):
    with shelve.open('data/problem_features.shelf') as shelf:
        pf = shelf['pf']
    num_tags = len(next(iter(pf.values()))['pf_tags'])

    users = {}
    cur = get_cur(db)
    cur.execute('SELECT COUNT(*) as count FROM records WHERE language=%s', ('C++', ))
    cnt = cur.fetchone()['count']
    x = np.empty((cnt, 9 + num_tags*3), dtype=np.float32)
    y = np.empty(cnt, dtype=np.int8)
    cur.close()

    cur = get_cur(db, named=True)
    cur.execute('SELECT owner, problem_id, result FROM records WHERE language=%s ORDER BY id', ('C++', ))
    cnt_rows = 0
    for record in tqdm(cur, total=cnt):
        if record['result'] in ['Compile Error', 'System Error', 'Unknown']:
            continue
        u = users.get(record['owner'], None)
        if u is None:
            u = users[record['owner']] = dict(
                num_submit=0,
                num_ac=0,
                num_tag_submit=np.zeros(num_tags, np.int32),
                num_tag_ac=np.zeros(num_tags, np.int32)
            )
        p = pf.get(record['problem_id'], None)
        if p is None:
            continue  # problem not exist
        
        if record['result'] == 'Accepted':
            label = 1
            u['num_ac'] += 1
            u['num_tag_ac'] += p['pf_tags']
        else:
            label = -1
        u['num_submit'] += 1
        u['num_tag_submit'] += p['pf_tags']
        x[cnt_rows] = calc_features(u, p)
        y[cnt_rows] = label
        cnt_rows += 1
    x = x[:cnt_rows, :]
    y = y[:cnt_rows]

    np.savez('data/data.npz', x=x, y=y)
    cur.close()


def run_makedata_recommend(db):
    with shelve.open('data/problem_features.shelf') as shelf:
        pf = shelf['pf']
    num_tags = len(next(iter(pf.values()))['pf_tags'])
    
    cur = get_cur(db)
    cur.execute('SELECT * FROM reports WHERE id=%s', (636, ))
    report = cur.fetchone()
    report_problem_list = tuple(map(int, report['problem_list'].strip().splitlines()))
    cur.execute('SELECT DISTINCT owner FROM records WHERE submit_datetime BETWEEN %s AND %s AND problem_id IN %s', (report['start_datetime'], report['end_datetime'], report_problem_list))
    report_student_list = [x['owner'] for x in cur]
    cur.close()

    users = {}
    x_train = np.empty((534586, 9 + num_tags*3), dtype=np.float32)
    y_train = np.empty(534586, dtype=np.int8)

    print('generate train data')
    cur = get_cur(db, named=True)
    cur.execute('SELECT owner, problem_id, result FROM records WHERE language=%s AND submit_datetime<%s ORDER BY id', ('C++', report['start_datetime']))
    cnt_train = 0
    for record in tqdm(cur):
        if record['result'] in ['Compile Error', 'System Error', 'Unknown']:
            continue
        u = users.get(record['owner'], None)
        if u is None:
            u = users[record['owner']] = dict(
                num_submit=0,
                num_ac=0,
                num_tag_submit=np.zeros(num_tags, np.int32),
                num_tag_ac=np.zeros(num_tags, np.int32)
            )
        p = pf.get(record['problem_id'], None)
        if p is None:
            continue  # problem not exist

        if record['result'] == 'Accepted':
            label = 1
            u['num_ac'] += 1
            u['num_tag_ac'] += p['pf_tags']
        else:
            label = -1
        u['num_submit'] += 1
        u['num_tag_submit'] += p['pf_tags']
        x_train[cnt_rows] = calc_features(u, p)
        y_train[cnt_train] = label
        cnt_train += 1

    x_train = x_train[:cnt_train, :]
    y_train = y_train[:cnt_train]
    user_list = [x for x in report_student_list if x in users]

    with shelve.open('data/recommend.shelf') as shelf:
        shelf['x_train'] = x_train
        shelf['y_train'] = y_train
        shelf['user_list'] = user_list
        shelf['user_features'] = users
        shelf['report_problem_list'] = report_problem_list
    cur.close()


def run_make_features(db):
    with shelve.open('data/problem_features.shelf') as shelf:
        pf = shelf['pf']
    num_tags = len(next(iter(pf.values()))['pf_tags'])

    cur = get_cur(db)
    cur.execute('SELECT COUNT(*) as count FROM records WHERE language=%s', ('C++', ))
    cnt = cur.fetchone()['count']
    cur.close()

    f = dict(
        uf_num_submit=np.empty(cnt, dtype=np.int16),
        uf_ac_rate=np.empty(cnt, dtype=np.float32),
        uf_tag_num_submit=np.empty((cnt, num_tags), dtype=np.int16),
        uf_tag_ac_rate=np.empty((cnt, num_tags), dtype=np.float32),
        problem_id=np.empty(cnt, dtype=np.int16),
        user_id=np.empty(cnt, dtype=np.int16),
        label=np.empty(cnt, dtype=np.bool_)
    )

    cur = get_cur(db, named=True)
    cur.execute('SELECT owner, problem_id, result FROM records WHERE language=%s ORDER BY id', ('C++', ))
    cnt_rows = 0
    users = {}
    for record in tqdm(cur, total=cnt):
        if record['result'] in ['Compile Error', 'System Error', 'Unknown']:
            continue
        u = users.get(record['owner'], None)
        if u is None:
            u = users[record['owner']] = dict(
                num_submit=0,
                num_ac=0,
                num_tag_submit=np.zeros(num_tags, np.int32),
                num_tag_ac=np.zeros(num_tags, np.int32),
                user_id=len(users)
            )
        problem_id = record['problem_id']
        p = pf.get(problem_id, None)
        if p is None:
            continue  # problem not exist

        f['user_id'][cnt_rows] = u['user_id']
        f['problem_id'][cnt_rows] = problem_id
        if record['result'] == 'Accepted':
            f['label'][cnt_rows] = 1
            u['num_ac'] += 1
            u['num_tag_ac'] += p['pf_tags']
        else:
            f['label'][cnt_rows] = 0
        u['num_submit'] += 1
        u['num_tag_submit'] += p['pf_tags']

        f['uf_num_submit'][cnt_rows] = u['num_submit']
        f['uf_ac_rate'][cnt_rows] = u['num_ac'] / u['num_submit'] if u['num_submit'] else 0
        f['uf_tag_num_submit'][cnt_rows] = u['num_tag_submit']
        f['uf_tag_ac_rate'][cnt_rows] = np_divide(u['num_tag_ac'].astype(np.float32), u['num_tag_submit'].astype(np.float32))
        cnt_rows += 1
    for k, v in f.items():
        f[k] = v[:cnt_rows]

    np.savez('data/features.npz', num_rows=cnt_rows, num_users=len(users), **f)
    cur.close()


def run_makedata_from_features():
    features = [
        'uf_num_submit',
        'uf_ac_rate',
        'uf_tag_num_submit',
        'uf_tag_ac_rate',
        'pf_num_submit',
        'pf_ac_rate',
        'pf_avg_lines',
        'pf_avg_bytes',
        'pf_avg_time',
        'pf_avg_mem',
        'pf_avg_score',
        'pf_tags'
    ]

    with shelve.open('data/problem_features.shelf') as shelf:
        pf = shelf['pf']
    sample_pf = next(iter(pf.values()))
    num_tags = len(sample_pf['pf_tags'])
    with np.load('data/features.npz') as data:
        f = dict(
            uf_num_submit=data['uf_num_submit'],
            uf_ac_rate=data['uf_ac_rate'],
            uf_tag_num_submit=data['uf_tag_num_submit'],
            uf_tag_ac_rate=data['uf_tag_ac_rate'],
            problem_id=data['problem_id'],
            user_id=data['user_id'],
            label=data['label'],
        )
        num_rows = int(data['num_rows'])
        num_users = int(data['num_users'])

    length = 0
    for name in features:
        if name.startswith('uf_'):
            sample = f[name][0]
        elif name.startswith('pf_'):
            sample = sample_pf[name]
        else:
            assert False
        if type(sample) is np.ndarray:
            length += len(sample)
        else:
            length += 1

    x = np.empty((num_rows, length), dtype=np.float32)
    y = np.empty(num_rows, dtype=np.int8)

    for row in tqdm(range(num_rows)):
        offset = 0
        p = pf[f['problem_id'][row]]
        for name in features:
            if name.startswith('uf_'):
                value = f[name][row]
            elif name.startswith('pf_'):
                value = p[name]
            else:
                assert False
            if type(value) is np.ndarray:
                flen = len(value)
            else:
                flen = 1
            x[row, offset:offset+flen] = value
            offset += flen
        y[row] = f['label'][row]
    np.savez('data/data.npz', x=x, y=y)



def main():
    db = get_db()
    # run_make_problem_features(db)
    # run_makedata(db)
    # run_makedata_recommend(db)
    # run_make_features(db)
    run_makedata_from_features()


if __name__ == '__main__':
    main()

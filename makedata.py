import re
import gzip
import shelve
import psycopg2
import psycopg2.extras
import multiprocessing
from tqdm import tqdm


def get_db():
    return psycopg2.connect('dbname=online_judge')


def get_cur(db):
    return db.cursor(cursor_factory=psycopg2.extras.DictCursor)


def parse_judge_message(msg):
    # return (verdict, time in ms, memory in kb)
    pattern = r'(.+?) \(Time: (\d+)ms, Memory: (\d+)kb\)'
    res = re.findall(pattern, msg)
    return list(map(lambda x: (x[0], int(x[1]), int(x[2])), res))


def pool_initializer():
    global db
    db = get_db()


def extract_single_problem_features(problem_id):
    global db
    cur = get_cur(db)

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

    with shelve.open('data/problem_features.shelf') as db:
        db['pf'] = pf


def run_makedata(db):
    cur = get_cur(db)
    with shelve.open('data/problem_features.shelf') as db:
        pf = db['pf']

    users = {}
    cur.execute('SELECT COUNT(*) as count FROM records WHERE language=%s', ('C++', ))
    cnt = cur.fetchone()['count']
    cnt_train = int(cnt * 0.8)
    f = gzip.open('data/data.txt.gz', 'wt')

    cur.execute('SELECT * FROM records WHERE language=%s', ('C++', ))
    for i, record in tqdm(enumerate(cur), total=cnt):
        if record['result'] in ['Compile Error', 'System Error', 'Unknown']:
            continue
        u = users.get(record['owner'], None)
        if u is None:
            u = users[record['owner']] = dict(num_submit=0, num_ac=0)
        p = pf.get(record['problem_id'], None)
        if p is None:
            continue  # problem not exist
        
        if record['result'] == 'Accepted':
            label = 1
            u['num_ac'] += 1
        else:
            label = -1
        u['num_submit'] += 1
        features = [
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

        f.write(str(label))
        for j, v in enumerate(features):
            f.write(' {}:{}'.format(j, v))
        f.write('\n')

    f.close()
        


def main():
    db = get_db()
    # run_make_problem_features(db)
    run_makedata(db)


if __name__ == '__main__':
    main()

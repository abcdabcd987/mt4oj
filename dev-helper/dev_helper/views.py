import random
import re
import psycopg2
import psycopg2.extras
from flask import request, redirect, session, url_for, render_template, g, jsonify
from dev_helper import app

def get_cur():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = psycopg2.connect(app.config['ONLINE_JUDGE_DB'])
    return db.cursor(cursor_factory=psycopg2.extras.DictCursor)


@app.teardown_appcontext
def teardown_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def parse_judge_message(msg):
    # return (verdict, time in ms, memory in kb)
    pattern = r'(.+?) \(Time: (\d+)ms, Memory: (\d+)kb\)'
    res = re.findall(pattern, msg)
    return list(map(lambda x: (x[0], int(x[1]), int(x[2])), res))


@app.route('/')
def get_homepage():
    return render_template('homepage.html')


@app.route('/problem/random')
def get_problem_random():
    cur = get_cur()
    cur.execute('SELECT id FROM problems')
    rows = cur.fetchall()
    problem_id = random.choice(rows)['id']
    return redirect(url_for('get_problem', problem_id=problem_id))


@app.route('/problem/<int:problem_id>')
def get_problem(problem_id):
    cur = get_cur()
    cur.execute('SELECT * FROM problems WHERE id=%s', (problem_id, ))
    problem = cur.fetchone()

    num_records = 0
    num_accepted = 0
    avg_code_lines = 0
    avg_code_bytes = 0
    avg_run_time = 0
    avg_run_mem = 0
    cur.execute('SELECT * FROM records WHERE problem_id=%s AND language=%s', (problem_id, 'C++'))
    for record in cur:
        num_records += 1
        if record['result'] != 'Accepted':
            continue
        num_accepted += 1
        avg_code_lines += len(record['submit_code'].splitlines())
        avg_code_bytes += len(record['submit_code'])
        testcases = parse_judge_message(record['judge_message'])
        avg_run_time += sum(map(lambda x: x[1], testcases))
        avg_run_mem += max(map(lambda x: x[1], testcases))
    if num_accepted:
        avg_code_lines /= num_accepted
        avg_code_bytes /= num_accepted
        avg_run_time /= num_accepted
        avg_run_mem /= num_accepted

    return render_template('problem.html',
                           problem=problem,
                           num_records=num_records,
                           num_accepted=num_accepted,
                           avg_code_lines=avg_code_lines,
                           avg_code_bytes=avg_code_bytes,
                           avg_run_time=avg_run_time,
                           avg_run_mem=avg_run_mem)


@app.route('/json/record/<int:record_id>.json')
def json_record(record_id):
    cur = get_cur()
    cur.execute('SELECT * FROM records WHERE id=%s', (record_id, ))
    record = dict(cur.fetchone())
    if record:
        record['ok'] = True
    else:
        record = {'ok': False}
    return jsonify(record)


@app.route('/json/problem/<int:problem_id>/random-accepted-record.json')
def json_problem_random_accepted_record(problem_id):
    cur = get_cur()
    cur.execute('SELECT id FROM records WHERE problem_id=%s AND language=%s AND result=%s ORDER BY RANDOM() LIMIT 1', (problem_id, 'C++', 'Accepted'))
    record = cur.fetchone()
    if record:
        return redirect(url_for('json_record', record_id=record['id']))
    return {'ok': False}

import shelve
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


def get_features(key):
    features = getattr(g, '_features', None)
    if features is None:
        features = g._features = {}
        with shelve.open('../data/problem_features.shelf') as db:
            features['pf'] = db['pf']
    return features[key]


@app.teardown_appcontext
def teardown_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


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
    pf = get_features('pf')[problem_id]

    return render_template('problem.html',
                           problem=problem,
                           pf=pf)


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

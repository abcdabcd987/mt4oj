from flask import Flask

app = Flask('mt4oj_dev_helper')
app.config['ONLINE_JUDGE_DB'] = 'dbname=online_judge'
app.config['TEMPLATES_AUTO_RELOAD'] = bool(app.debug)
app.jinja_env.auto_reload = bool(app.debug)

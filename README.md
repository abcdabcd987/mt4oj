# mt4oj

Download the database at <https://www.dropbox.com/s/hgjd7im3sc0kkly/online_judge.pg_dump?dl=0>.

```
sudo -u postgres  createuser -s $USER
sudo -u postgres  createdb online_judge
sudo -u postgres  pg_restore --no-owner --role $USER -d online_judge online_judge.pg_dump
python3 makedata.py gen_problem_features
python3 makedata.py gen_user_features
python3 makedata.py concat_features
python3 run_rnn.py
python3 rl_a3c.py
```

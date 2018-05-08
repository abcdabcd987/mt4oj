# mt4oj

## Data

Download the database at <https://www.dropbox.com/s/mc2qaoxt46wrngn/mt4oj_online_judge.pg_dump?dl=0>.

```
sudo -u postgres createuser -s $USER
createdb online_judge
pg_restore --no-owner --role $USER -d online_judge mt4oj_online_judge.pg_dump
```

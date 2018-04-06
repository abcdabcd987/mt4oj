import re
import requests


def main():
    tags = set()
    for page_no in range(1, 43):
        print('============= Page', page_no)
        url = 'http://codeforces.com/problemset/page/{}'.format(page_no)
        r = requests.get(url)
        matches = re.findall(r'"\/problemset\/tags\/(.*?)"', r.text)
        tags = tags.union(matches)
        print(tags)


if __name__ == '__main__':
    main()


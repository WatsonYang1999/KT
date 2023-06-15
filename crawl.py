import requests


r = requests.get(url="https://www.douban.com/",verify=False)

print(r)
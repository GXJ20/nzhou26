## This scripts download every entry on empiar and store its metadata in json format

import requests
import json

# currently the largest empiar id is 10983, as more dataset deposit to empiar, this range 
# can be edited
for i in range(10001,10983):
    # url of rest api of empiar
    url = f'https://www.ebi.ac.uk/empiar/api/entry/{i}/'
    page = requests.get(url)
    # parse info into json
    with open(f'json/{i}.json', 'w') as fp:
        json.dump(page.json(), fp)
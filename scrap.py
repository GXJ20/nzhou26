# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import requests
import json
import pandas as pd


# %%
for i in range(10001,10839):
    print(i)
    url = f'https://www.ebi.ac.uk/empiar/api/entry/{i}/'
    page = requests.get(url)
    with open(f'empiar_db_json/{i}.json', 'w') as fp:
        json.dump(page.json(), fp)



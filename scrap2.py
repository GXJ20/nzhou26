# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import requests
import json
import pandas as pd
import pathlib


# %%
titles = []
empiar_ids = []
dataset_sizes = []
emdb_ids = []
best_resolutions = []
for item in list(pathlib.Path('empiar_db_json').glob('*.json')):
    data = json.load(open(item))
    first_value = list(data.values())[0]
    if 'picked particles' in str(first_value):
        if 'EMD-' in str(first_value):
            try:
                empiar_id = list(data.keys())[0][-5:]
                
                print(f"title: {first_value['title']}")
                print(f"empiar id: {empiar_id}")
                print(f"dataset_size: {first_value['dataset_size']}")
                print(f"EMDB: {first_value['cross_references']}")
                
                resolutions = []
                for emd in first_value['cross_references']:
                    emdb_id = emd.split('-')[1]
                    url = f'https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}'
                    page = requests.get(url)
                    emdb_info = page.json()
                    resolution = emdb_info['structure_determination_list']['structure_determination'][0]['image_processing'][0]['final_reconstruction']['resolution']['valueOf_']
                    resolutions.append(float(resolution))
                resolutions = sorted(resolutions)
                highest_resolution = resolutions[0]
                print(f'highest_resolution: {highest_resolution}')
                print()
                titles.append(first_value['title'])
                empiar_ids.append(empiar_id)
                dataset_sizes.append(first_value['dataset_size'])
                emdb_ids.append(first_value['cross_references'])
                best_resolutions.append(highest_resolution)
            except:
                print(f'not found in {item}')


# %%
df =pd.DataFrame({'empiar_ids':empiar_ids,
                'title':titles,
                'dataset_sizes':dataset_sizes,
                'emdb_ids':emdb_ids,
                'best_resolutions':best_resolutions})


# %%
df.to_csv('useful_data.csv')


# %%




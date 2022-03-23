# %%
import requests
import json
import pandas as pd
import pathlib

## This script read all json file and save useful information only to a single csv file

# %%
# create empty lists that are ready to take info(metadata) from dataset in empiar
titles = []
empiar_ids = []
dataset_sizes = []
emdb_ids = []
best_resolutions = []

# iterate through every json file
for item in list(pathlib.Path('json').glob('*.json')):
    # load json file
    data = json.load(open(item))
    
    first_value = list(data.values())[0]
    # check if this dataset contains particle coordinates
    if 'picked particles' in str(first_value):
        if 'EMD-' in str(first_value):
            try:
                empiar_id = list(data.keys())[0][-5:]

                # print out other metadata
                print(f"title: {first_value['title']}")
                print(f"empiar id: {empiar_id}")
                print(f"dataset_size: {first_value['dataset_size']}")
                print(f"EMDB: {first_value['cross_references']}")

                # find final resolutions of this dataset
                resolutions = []
                
                for emd in first_value['cross_references']:
                    # find respective emdb ids 
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
                # append all of them into the big lists
                titles.append(first_value['title'])
                empiar_ids.append(empiar_id)
                dataset_sizes.append(first_value['dataset_size'])
                emdb_ids.append(first_value['cross_references'])
                best_resolutions.append(highest_resolution)
            except:
                print(f'not found in {item}')


# export them into pandas dataframe
df =pd.DataFrame({'empiar_ids':empiar_ids,
                'title':titles,
                'dataset_sizes':dataset_sizes,
                'emdb_ids':emdb_ids,
                'best_resolutions':best_resolutions})


# save it to csv
df.to_csv('useful_data.csv')
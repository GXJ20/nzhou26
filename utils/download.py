import os
import pandas as pd
# you should detach this script in the background using a screen session.

# every possible useful dataset are store in the csv file
df = pd.read_csv('trimmed_empiar_data.csv', index_col=0)
# select out ones are not downloaded
df = df[df['to_be_download']]
added_size = 0
# given an available space, in GB
avail_size = 2*1024

# there are three possible source to download, the first one in china is stable but missing
# newer datasets, the second one in UK is not stable but contains all the datasets, the third one
# in japan contains all the datasets, and is relatively stable.
#database = 'ftp://ftp.empiar-china.org/'
#database = 'ftp://ftp.ebi.ac.uk/empiar/world_availability/'
database = 'ftp://empiar.pdbj.org/pub/empiar/archive/'
for index, item in df.iterrows():
    added_size += item['converted_size']
    if added_size < avail_size:
        empiar_id = item['empiar_ids']
        print(f"downloading {empiar_id}")
        # mwget is a downloading tool that is multi-process based and support break-point continuely
        os.system(f'mwget -c -m {database}{empiar_id}')
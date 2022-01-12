# %%
import pathlib
import numpy as np

def save_npz_compressed(npy_dir, dataset_name, my_dict):
    print(f'Saving data in {npy_dir}')
    for item in npy_dir.glob('*.npy'):
        my_dict[f'{dataset_name}--{item.name}'] = np.load(item)

# %%
data_for_train = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/data_for_training/'
raw_dict = {}
for path in pathlib.Path(data_for_train).glob('*/raw'):
    save_npz_compressed(path, path.parent.name, raw_dict)
np.savez_compressed(f'{data_for_train}/raw.npz', **raw_dict)
label_dict = {}
for path in pathlib.Path(data_for_train).glob('*/label'):
    save_npz_compressed(path, path.parent.name, label_dict)
np.savez_compressed(f'{data_for_train}/label.npz', **label_dict)

    


# %%

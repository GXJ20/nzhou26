# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import starfile
import mrcfile
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

df = pd.read_json('data_for_training.json')
dataset_name = 'cleaved_rhs'
df = df[df['dataset_name'] == dataset_name].squeeze()
map_path = df['map_path']
mask_path = df['mask_path']
meta_star = df['meta_data_path']
data_dir = df['raw_data_dir']

meta = starfile.read(meta_star)
def check_len():
    df_len_check = pd.read_json('data_for_training.json')
    for index, row in df_len_check.iterrows():
        meta_len = len(starfile.read(row['meta_data_path']))
        if meta_len < row['number_of_particles_to_process'] :
            print(f"{row['dataset_name']} exceeds length")
check_len()


# %%
def mask_and_proj(path1, path2):
    projection_command = f"relion_project --i {path1} --o {path2}        --rot {meta['rlnAngleRot'][idx]} --tilt {meta['rlnAngleTilt'][idx]} --psi {meta['rlnAnglePsi'][idx]}        --xoff {meta['rlnOriginX'][idx]}, --yoff {meta['rlnOriginX'][idx]} --ctf"
    os.system(projection_command)


# %%


idx = 3
raw_path = meta['rlnImageName'][idx].split('@')[-1]
particle_idx = int(meta['rlnImageName'][idx].split('@')[0]) - 1
raw_name = raw_path.split('/')[-1]
raw_path = f"{data_dir}{raw_name}"

print(particle_idx)
print(raw_path)
with mrcfile.open(raw_path) as mrc:
    raw = mrc.data[particle_idx]
mask_proj_name = f'{particle_idx}@{raw_name}'
mask_proj_path = f'tmp/mask_{mask_proj_name}'

map_proj_name = f'{particle_idx}@{raw_name}'
map_proj_path = f'tmp/map_{map_proj_name}'

mask_and_proj(mask_path, mask_proj_path)

mask_and_proj(map_path, map_proj_path)

with mrcfile.open(mask_proj_path) as mrc:
    mask_proj = mrc.data
with mrcfile.open(map_proj_path) as mrc:
    map_proj = mrc.data

mask_proj.setflags(write=1)
threshold =10
number_pixel_to_extend = 5
len_of_fill = number_pixel_to_extend*2 +1
mask_proj[mask_proj <= threshold] = 2
mask_proj[mask_proj > threshold] = 1
fil = np.full((len_of_fill,len_of_fill), -1)
fil[number_pixel_to_extend,number_pixel_to_extend] = len_of_fill*len_of_fill-1
contour = convolve(mask_proj, fil, mode='reflect') > 0
mask_proj[contour == True] = 3
mask_proj = mask_proj.astype('uint8')
with mrcfile.open(mask_proj_path) as mrc:
    mask_proj_raw = mrc.data


# %%
fig, ax = plt.subplots(3,2,figsize=(10,10))

ax[0,0].imshow(raw, cmap='gray')
ax[0,0].set_title('raw')

ax[0,1].imshow(gaussian_filter(raw, sigma=5), cmap='gray')
ax[0,1].set_title('lowpass')

ax[1,0].imshow(map_proj, cmap='gray')
ax[1,0].set_title('map projection')

ax[1,1].imshow(mask_proj_raw, cmap='gray')
ax[1,1].set_title('mask projection')

ax[2,0].imshow(mask_proj, cmap='gray')
ax[2,0].set_title('mask segmented')

ax[2,1].imshow(gaussian_filter(raw, sigma=5), cmap='gray')
ax[2,1].imshow(mask_proj, alpha=0.7)
ax[2,1].set_title('mask applied to lowpass')
plt.show()

# %%
import tensorflow as tf
import random as rn
seed = 131

# tf.random.set_seed(seed)
# #tf.kera.preprocessing.image.
# np.random.seed(seed)
# rn.seed(seed)

shift = 0.3
raw_shift = np.stack((raw,)*3, axis=-1)

#raw_shift = tf.image.random_crop(raw_shift, size = seed=seed)
h, w = raw_shift.shape[0], raw_shift.shape[1]
tx = np.random.uniform(-shift, shift)*h
ty = np.random.uniform(-shift, shift)*w
raw_shift = tf.keras.preprocessing.image.apply_affine_transform(raw_shift, tx,ty , channel_axis=2, fill_mode='nearest') 
mask_proj_shift = np.expand_dims(mask_proj, 2)
mask_proj_shift = tf.keras.preprocessing.image.apply_affine_transform(mask_proj_shift, tx,ty , channel_axis=2, fill_mode='nearest') 

fig, ax = plt.subplots(4,1,figsize=(10,10))

ax[0].imshow(raw_shift, cmap='gray')
ax[1].imshow(gaussian_filter(raw_shift, sigma=5), cmap='gray')
ax[2].imshow(mask_proj_shift, cmap='gray')
ax[3].imshow(gaussian_filter(raw, sigma=5), cmap='gray')
ax[3].imshow(mask_proj_shift, alpha=0.7)



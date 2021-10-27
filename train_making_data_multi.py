# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import sys
import starfile
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

# %%
meta_refine026 = starfile.open('20s_10025/Refine3D/job026/run_data.star')
meta_recenter = starfile.open('20s_10025/Extract/job011/particles.star')
image_idx = int(sys.argv[1])
imageName = meta_recenter['rlnImageName'][image_idx]
refine026 = meta_refine026[meta_refine026['rlnImageName'] == imageName]#[['rlnAngleRot', 'rlnAngleTilt','rlnAnglePsi', 'rlnOriginX', 'rlnOriginY']].iloc[0]

# %%
particle_idx = int(imageName.split("@")[0]) -1
image_path = imageName.split('@')[1]
with mrcfile.open(f'20s_10025/{image_path}') as mrc:
    mrc_stack_recenter = mrc.data[particle_idx]
    mrc_stack_recenter_gs = gaussian_filter(mrc_stack_recenter, sigma=5)
plt.imsave(f'data_multi/raw/{particle_idx}@{image_path.split("/")[-1]}.png', mrc_stack_recenter, cmap='gray')
plt.imsave(f'data_multi/lp/{particle_idx}@{image_path.split("/")[-1]}.png', mrc_stack_recenter_gs, cmap='gray')
# %%
mask_path = "20s_10025/MaskCreate/job029/mask.mrc"
refine_path = "20s_10025/Refine3D/job026/run_class001.mrc"
output_path_mask = f'data_multi/proj/{sys.argv[1]}.mrc'
output_path_refine = "refine_proj.mrc"
rot = float(refine026['rlnAngleRot'])
tilt = float(refine026['rlnAngleTilt'])
psi = float(refine026['rlnAnglePsi'])
xoff = float(refine026['rlnOriginX'])
yoff = float(refine026['rlnOriginY'])

# %%
print(f'relion_project             --i {mask_path} --o {output_path_mask}            --rot {rot} --tilt {tilt}            --psi {psi} --xoff {xoff}            --yoff {yoff}')
os.system(f'relion_project             --i {mask_path} --o {output_path_mask}            --rot {rot} --tilt {tilt}            --psi {psi} --xoff {xoff}            --yoff {yoff}')

# %%
threshold = 10
with mrcfile.open(output_path_mask) as mrc:
    mask_proj = mrc.data
mask_proj.setflags(write=1)
mask_proj[mask_proj <= threshold] = 2
mask_proj[mask_proj > threshold] = 1

fil = np.full((11,11), -1)
fil[5,5] = 11*11-1
contour = convolve(mask_proj, fil, mode='reflect') > 0
mask_proj[contour == True] = 3
mask_proj = mask_proj.astype('uint8')

plt.imsave(f'data_multi/mask/{particle_idx}@{image_path.split("/")[-1]}.png', mask_proj, cmap='gray')




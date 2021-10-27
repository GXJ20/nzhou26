# This script is used to generate figures that contains paritcle raw image,
# low-pass filtered image, projection, and masks
import starfile
import mrcfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifftn
from scipy.ndimage import gaussian_filter
import os
import sys


# %%
meta_2d = starfile.open('20s_10025/Class2D/job006/run_it025_data.star')
meta_raw = starfile.open('20s_10025/Extract/job005/particles.star')
meta_3d_c1 = starfile.open('20s_10025/Class3D/job007/run_it025_data.star')
meta_3d_d7 = starfile.open('20s_10025/Class3D/job008/run_it025_data.star')
meta_3d_refine = starfile.open('20s_10025/Refine3D/job009/run_data.star')
meta_recenter = starfile.open('20s_10025/Extract/job010/particles.star')


# %%
image_idx = int(sys.argv[1])
imageName = meta_raw['rlnImageName'][image_idx]
print(imageName)
raw = meta_raw[meta_raw['rlnImageName'] == imageName][['rlnAngleRot', 'rlnAngleTilt','rlnAnglePsi']].iloc[0]
class_2d = meta_2d[meta_2d['rlnImageName'] == imageName][['rlnAngleRot', 'rlnAngleTilt','rlnAnglePsi', 'rlnClassNumber']].iloc[0]
c1 = meta_3d_c1[meta_3d_c1['rlnImageName'] == imageName][['rlnAngleRot', 'rlnAngleTilt','rlnAnglePsi', 'rlnOriginX', 'rlnOriginY']].iloc[0]
d7 = meta_3d_d7[meta_3d_d7['rlnImageName'] == imageName][['rlnAngleRot', 'rlnAngleTilt','rlnAnglePsi', 'rlnOriginX', 'rlnOriginY']].iloc[0]
refine = meta_3d_refine[meta_3d_refine['rlnImageName'] == imageName][['rlnAngleRot', 'rlnAngleTilt','rlnAnglePsi', 'rlnOriginX', 'rlnOriginY']].iloc[0]
recenter = meta_recenter[meta_raw['rlnImageName'] == imageName][['rlnAngleRot', 'rlnAngleTilt','rlnAnglePsi','rlnImageName']].iloc[0]


# %%
print(raw)
print(class_2d)
print(c1)
print(d7)
print(refine)
print(recenter)
print(recenter['rlnImageName'].split('@')[1])


# %%
def relion_projection(map, output,rot,tilt,psi, xoff, yoff):
    os.system(f'relion_project --i {map} --o {output} --rot {rot} --tilt {tilt} --psi {psi} --xoff {xoff}, --yoff {yoff}')
def relion_lowpass(img, output, lp):
    os.system(f'relion_image_handler --i {img} --o {output} --lowpass {lp}')


# %%
relion_lowpass(f'20s_10025/{imageName.split("@")[1]}', 'lowpass.mrcs', 20)
relion_projection('20s_10025/Class3D/job007/run_it025_class001.mrc', 'projc1.mrc', c1['rlnAngleRot'], c1['rlnAngleTilt'], c1['rlnAnglePsi'], c1['rlnOriginX'], c1['rlnOriginY'])
relion_projection('20s_10025/Class3D/job008/run_it025_class001.mrc', 'projd7.mrc', d7['rlnAngleRot'], d7['rlnAngleTilt'], d7['rlnAnglePsi'], d7['rlnOriginX'], d7['rlnOriginY'])
relion_projection('20s_10025/Refine3D/job009/run_class001.mrc', 'projRefine.mrc', refine['rlnAngleRot'], refine['rlnAngleTilt'], refine['rlnAnglePsi'], refine['rlnOriginX'], refine['rlnOriginY'])
relion_lowpass(f'20s_10025/{recenter["rlnImageName"].split("@")[1]}', 'lowpass_recenter.mrcs', 20)


# %%
particle_idx = int(imageName.split("@")[0]) -1
class_number = int(class_2d['rlnClassNumber'])-1
with mrcfile.open(f'20s_10025/{imageName.split("@")[1]}') as mrc:
    mrc_stack = mrc.data[particle_idx]
    mrc_stack_gs = gaussian_filter( mrc_stack, sigma=5)
with mrcfile.open(f'lowpass.mrcs') as mrc:
    mrc_stack_lp = mrc.data[particle_idx]
with mrcfile.open('20s_10025/Class2D/job006/run_it025_classes.mrcs') as mrc:
    mrc_class = mrc.data[class_number]
with mrcfile.open('projc1.mrc') as mrc:
    mrc_projc1 = mrc.data
with mrcfile.open('projd7.mrc') as mrc:
    mrc_projd7 = mrc.data
with mrcfile.open('projRefine.mrc') as mrc:
    mrc_projRefine = mrc.data
with mrcfile.open(f'20s_10025/{recenter["rlnImageName"].split("@")[1]}') as mrc:
    mrc_stack_recenter = mrc.data[particle_idx]
    mrc_stack_recenter_gs = gaussian_filter(mrc_stack_recenter, sigma=5)
with mrcfile.open(f'lowpass_recenter.mrcs') as mrc:
    mrc_stack_recenter_lp = mrc.data[particle_idx]


# %%
print(imageName)
plt.figure(figsize=(10,10))
ax1 = plt.subplot(4,4,1)
ax1.set_title('raw_particle')
plt.imshow(mrc_stack, cmap='gray')
ax1.axis('off')

ax2 = plt.subplot(4,4,2)
ax2.set_title('particle_lp_filter_relion')
plt.imshow(mrc_stack_lp, cmap='gray')
ax2.axis('off')

ax3 = plt.subplot(4,4,3)
ax3.set_title('particle_gaussian_filter')
plt.imshow(mrc_stack_gs, cmap='gray')
ax3.axis('off')

ax4 = plt.subplot(4,4,5)
ax4.set_title('class2D')
plt.imshow(mrc_class, cmap='gray')
ax4.axis('off')

ax5 = plt.subplot(4,4,6)
ax5.set_title('class3D_C1_projection')
plt.imshow(mrc_projc1, cmap='gray')
ax5.axis('off')

ax6 = plt.subplot(4,4,7)
ax6.set_title('class3D_D7_projection')
plt.imshow(mrc_projd7, cmap='gray')
ax6.axis('off')

ax7 = plt.subplot(4,4,8)
ax7.set_title('refine_D7_projection')
plt.imshow(mrc_projRefine, cmap='gray')
ax7.axis('off')

ax8 = plt.subplot(4,4,9)
ax8.set_title('raw_particle_recenter')
plt.imshow(mrc_stack_recenter, cmap='gray')
ax8.axis('off')

ax9 = plt.subplot(4,4,10)
ax9.set_title('recenter_lp_relion')
plt.imshow(mrc_stack_recenter_lp, cmap='gray')
ax9.axis('off')

ax10 = plt.subplot(4,4,11)
ax10.set_title('recenter_gs')
plt.imshow(mrc_stack_recenter_gs, cmap='gray')
ax10.axis('off')

plt.savefig(f'more_particle_images/{image_idx}.png')

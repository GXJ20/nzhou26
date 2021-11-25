# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import pathlib
import random
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
from particle import *
import mrcfile

data_dir = 'data_for_training'
pred_dir = '*'
base_model_name = '66.64--150000--particle_segmentation-2021-11-24'

img_size = (256, 256)
num_classes = 3
batch_size = 32
data_aug_fold = 4

data_paths = pathlib.Path(data_dir)

input_img_paths = list(data_paths.glob(f"{pred_dir}/raw/*"))

input_img_paths = sorted(input_img_paths, key=os.path.basename)

target_img_paths = list(data_paths.glob(f"{pred_dir}/label/*"))
target_img_paths = sorted(target_img_paths, key=os.path.basename)

class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)
model = tf.keras.models.load_model(f"models/{base_model_name}.h5", custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])
history = pd.read_csv(f'models/{base_model_name}-history.csv')
plt.plot(history['val_loss'], label='val_loss')
plt.plot(history['loss'], label = 'loss')
plt.plot(history['val_updated_mean_io_u'], label='val_mean_IOU')
plt.plot(history['updated_mean_io_u'], label = 'mean_IOU')

plt.legend()
plt.show()
#plt.savefig('figures/train_history.png')


# %%


val_samples = int(len(input_img_paths)//5)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

val_gen = particles(batch_size, img_size, val_input_img_paths[:64], val_target_img_paths[:64])
val_preds = model.predict(val_gen)
print(len(val_preds))


# %%

fig,ax = plt.subplots(4,4,figsize=(10,10))
random.seed(10)
for i in range(0,4):
    pred_idx = random.randint(0,len(val_preds)-1)
    raw = np.load(val_input_img_paths[pred_idx])
    true_mask = np.load(val_target_img_paths[pred_idx])
    pred_mask = np.argmax(val_preds[pred_idx], axis=-1)
    #pred_mask = np.expand_dims()
    ax[0,i].imshow(raw, cmap='gray')
    ax[0,i].set_title(f'{pred_idx}:raw')
    ax[1,i].imshow(gaussian_filter(raw, sigma=5), cmap='gray')
    ax[1,i].set_title('low pass filtered')
    ax[2,i].imshow(true_mask, cmap='gray')
    ax[2,i].set_title('true mask')
    ax[3,i].imshow(pred_mask, cmap='gray')
    ax[3,i].set_title('predicted mask')
    [axi.set_axis_off() for axi in ax.ravel()]
plt.show()
#fig.savefig('figures/predictions_in_dataset.png')


# %%
test_idx = 0
test_dir = '/storage_data/gao_xijie/6P_fab_ACE2_0524/Extract/job056/goodMRC'
test_paths = list(pathlib.Path(test_dir).glob('*.mrcs'))[:5]
os.system('rm tmp/*')
for item in test_paths:
    with mrcfile.open(item) as mrc:
        for i in range(len(mrc.data)):
            np.save(f'tmp/{i+1}@@{item.name[:-1]}.npy', mrc.data[i])
test_paths = list(pathlib.Path('tmp').glob('*.npy'))
test_gen = inference_particles(4,img_size,input_img_paths=test_paths)
test_preds = model.predict(test_gen)
print(len(test_paths))


# %%

fig,ax = plt.subplots(3,3,figsize=(10,10))
#random.seed(10)
for i in range(0,3):
    #random.seed(i)
    pred_idx = random.randint(0,len(test_paths))
    raw = np.load(test_paths[pred_idx])
    pred_mask = np.argmax(test_preds[pred_idx], axis=-1)
    ax[0,i].imshow(raw, cmap='gray')
    ax[0,i].set_title(f'{pred_idx}:raw')
    ax[1,i].imshow(gaussian_filter(raw, sigma=5), cmap='gray')
    ax[1,i].set_title('low pass filtered')
    ax[2,i].imshow(pred_mask, cmap='gray')
    ax[2,i].set_title('predicted mask')
    [axi.set_axis_off() for axi in ax.ravel()]
#fig.savefig('figures/predictions_in_gpr158.png')
plt.show()

# %%

# %%

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

img_size = (256, 256)
num_classes = 3
batch_size = 32
data_aug_fold = 4

data_paths = pathlib.Path(data_dir)
input_img_paths = list(data_paths.glob(f"{pred_dir}/raw/*"))
input_img_paths = sorted(input_img_paths, key=os.path.basename)
target_img_paths = list(data_paths.glob(f"{pred_dir}/label/*"))
target_img_paths = sorted(target_img_paths, key=os.path.basename)
val_samples = int(len(input_img_paths)//5)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

test_dir = 'data_for_training/test'
test_paths = pathlib.Path(test_dir)
test_input_img_paths = list(test_paths.glob(f"*/raw/*"))
test_input_img_paths = sorted(test_input_img_paths, key=os.path.basename)
test_target_img_paths = list(test_paths.glob(f"*/label/*"))
test_target_img_paths = sorted(test_target_img_paths, key=os.path.basename)


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

def display_pred(input_paths, target_paths, pred, ran):
    fig,ax = plt.subplots(4,4,figsize=(10,10))
    if not ran :
        random.seed(10)
    for i in range(0,4):
        pred_idx = random.randint(0,len(pred)-1)
        raw = np.load(input_paths[pred_idx])
        true_mask = np.load(target_paths[pred_idx])
        pred_mask = np.argmax(pred[pred_idx], axis=-1)
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
# %%
base_model_name = '67.58--290000--particle_segmentation-2021-12-15'
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

val_gen = particles(batch_size, img_size, val_input_img_paths[:256], val_target_img_paths[:256], val=True)
val_preds = model.predict(val_gen)
val_loss, val_iou = model.evaluate(val_gen)
display_pred(val_input_img_paths, val_target_img_paths, val_preds, False)
# %%

test_gen = particles(batch_size, img_size, test_input_img_paths[:1024], test_target_img_paths[:1024], val=True)
test_preds = model.predict(test_gen)
test_loss, test_iou = model.evaluate(test_gen)
display_pred(test_input_img_paths, test_target_img_paths, test_preds, False)


# %%
infer_dir = '/storage_data/gao_xijie/6P_fab_ACE2_0524/Extract/job056/goodMRC'
infer_paths = list(pathlib.Path(infer_dir).glob('*.mrcs'))[:5]
os.system('rm tmp/*')
for item in infer_paths:
    with mrcfile.open(item) as mrc:
        for i in range(len(mrc.data)):
            np.save(f'tmp/{i+1}@@{item.name[:-1]}.npy', mrc.data[i])
infer_paths = list(pathlib.Path('tmp').glob('*.npy'))
infer_gen = inference_particles(4,img_size,input_img_paths=infer_paths)
infer_preds = model.predict(infer_gen)
print(len(infer_paths))


# %%

fig,ax = plt.subplots(3,3,figsize=(10,10))
#random.seed(10)
for i in range(0,3):
    #random.seed(i)
    pred_idx = random.randint(0,len(infer_paths))
    raw = np.load(infer_paths[pred_idx])
    pred_mask = np.argmax(infer_preds[pred_idx], axis=-1)
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

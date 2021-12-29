# %%
import tensorflow as tf
import numpy as np
import pathlib
import random
import os
import pandas as pd
from particle import *
import cv2
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
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]
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
# %%
base_model_name = '67.58--290000--particle_segmentation-2021-12-15'
model = tf.keras.models.load_model(f"models/{base_model_name}.h5", custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])

def save_pred_and_iou(start_idx, end_idx):
  for idx in range(start_idx,end_idx):
      pred_idx = idx % batch_size
      pred = np.argmax(val_preds[pred_idx],axis=-1)
      if not os.path.exists(f'preds/{val_target_img_paths[idx].name}'):        
        np.save(f'preds/{val_target_img_paths[idx].name}', pred)
      names.append(val_target_img_paths[idx].name)
      true = np.load(val_target_img_paths[idx]) -1
      pred = cv2.resize(pred, dsize=true.shape, interpolation=cv2.INTER_NEAREST)
      m = tf.keras.metrics.MeanIoU(num_classes=3)
      m.update_state(pred,true)
      m = m.result().numpy()
      ious.append(m)
      print("{0:.2%}".format(idx/total_len))
ious = []
names = []
num_of_particles = 50000
batch_to_pred = num_of_particles//batch_size
total_len = batch_to_pred * batch_size
for i in range(0,batch_to_pred):
  start = i*batch_size
  end = start + batch_size
  val_gen = particles(batch_size, img_size, val_input_img_paths[start:end], val_target_img_paths[start:end], val=True)
  val_preds = model.predict(val_gen)
  save_pred_and_iou(start,end)
df = pd.DataFrame({
    'iou':ious,
    'name':names})
df.to_csv('preds/preds_iou.csv')
# %%


# %%

# %%

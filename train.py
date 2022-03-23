# %%
import os
import random
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pathlib
import sys
from datetime import date
from particle import *
import pandas as pd
from helper import UpdatedMeanIoU
import models
import datetime
tf.random.set_seed(42)
mirrored_strategy = tf.distribute.MirroredStrategy()

data_dir = '/ssd/data_for_training/'
output_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/new/'
img_size = (256, 256)
num_classes = 3



# %%

class Train():
  def __init__(self, model_name, num_to_use, fine_tune, fold=0, batch_size=16 ):
    self.model_name = model_name
    self.num_to_use = num_to_use
    self.fold = fold
    self.batch_size = batch_size
    self.fine_tune = fine_tune
  def train(self):
    train_gen, val_gen, test_gen = self.data_gen(self.num_to_use)
    
    with mirrored_strategy.scope():
      model = self.create_model(model_name=self.model_name)
    # tf.keras.utils.plot_model(model, to_file=f'{self.model_name}.png', show_shapes=True)
    # print('model plot saved!')

    tmp_model_name = f"{self.model_name}--{self.fine_tune}--{date.today()}.h5"
    # set up tensorboard to keep track of this training
    log_dir = f"../data_particleSeg/logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}{self.model_name}{self.fine_tune}"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{output_dir}{tmp_model_name}", save_best_only=True),
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
	      #tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 9))
    ]
    steps_per_epoch = train_gen.__len__()
    validation_steps = val_gen.__len__()
    history = model.fit(
        train_gen,
        callbacks=callbacks,
        epochs=30, steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps)
    hist_df = pd.DataFrame(history.history)
    
    model = tf.keras.models.load_model(f"{output_dir}{tmp_model_name}", custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})

    loss, iou = model.evaluate(test_gen)
    iou = "{0:.2%}".format(iou)[:-1]
    permanet_model_name = f"{iou}--{self.num_used}--{tmp_model_name}"
    hist_csv_file = f'{output_dir}{permanet_model_name}-history.csv'
    os.system(f"mv {output_dir}{tmp_model_name} \
          {output_dir}{permanet_model_name}")
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

  def create_model(self,model_name):
    if model_name == 'custom':
      return models.custom_unet()
    else:
      return models.pretrained_model(model_name, fine_tune_at=100)
  # this function controls data generation
  def data_gen(self, num_to_use):
    data_paths = pathlib.Path(data_dir)
    input_img_paths = list(data_paths.glob("*/raw/*.npy"))
    input_img_paths = sorted(input_img_paths, key=os.path.basename)
    target_img_paths = list(data_paths.glob("*/label/*.npy"))
    target_img_paths = sorted(target_img_paths, key=os.path.basename)
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    
    if num_to_use != -1:
      input_img_paths = input_img_paths[:num_to_use]
      target_img_paths = target_img_paths[:num_to_use]
    self.num_used = len(input_img_paths)
    # Split our img paths into a training and a validation set
    test_samples = int(len(input_img_paths)//5)
    val_samples = int(len(input_img_paths)//5*4//5)
    train_input_img_paths = input_img_paths[:-test_samples]
    train_target_img_paths = target_img_paths[:-test_samples]
    test_input_img_paths = input_img_paths[-test_samples:]
    test_target_img_paths = target_img_paths[-test_samples:]

    val_input_img_paths = train_input_img_paths[-val_samples:]
    val_target_img_paths = train_target_img_paths[-val_samples:]
    train_input_img_paths = train_input_img_paths[:-val_samples]
    train_target_img_paths = train_target_img_paths[:-val_samples]
    print(len(train_input_img_paths))
    print(len(val_input_img_paths))
    print(len(test_input_img_paths))
    train_gen = particles(self.batch_size, img_size, train_input_img_paths, train_target_img_paths,fold=self.fold)
    val_gen = particles(self.batch_size, img_size, val_input_img_paths, val_target_img_paths,fold=self.fold)
    test_gen = particles(self.batch_size, img_size, test_input_img_paths, test_target_img_paths, fold=self.fold)
    return train_gen, val_gen, test_gen

base_models = [
  #'custom',
  #'DenseNet121',
  #'DenseNet169',
  #'DenseNet201',
  'EfficientNetB0',
  'ResNet101'
]
if __name__ =='__main__':
  if sys.argv[1] == 'all':
    for model in base_models:
      new_train = Train(model, 40000, batch_size=32)
      new_train.train()
  elif sys.argv[1] == 'fine_tune':
    for fine_tune in [0,50,100,200,300,500]:
      new_train = Train('DenseNet169',40000, batch_size=16, fine_tune=fine_tune)
      new_train.train()
  else:
    new_train = Train(sys.argv[1], 40000, batch_size=16)
    new_train.train()

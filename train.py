# %%
import os
import random
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pathlib
from datetime import date
from particle import *
from tensorflow_examples.models.pix2pix import pix2pix
import pandas as pd
from tensorflow.keras import layers

mirroted_strategy = tf.distribute.MirroredStrategy()

data_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/data_for_training/'
output_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/'
img_size = (256, 256)
num_classes = 3
batch_size = 16

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

class Train():
  def __init__(self, model_name, num_to_use):
    self.model_name = model_name
    self.num_to_use = num_to_use
  def train(self):
    train_gen, val_gen, test_gen = self.data_gen(self.num_to_use)
    with mirroted_strategy.scope():
      model = self.create_model(model_name=self.model_name)
    tmp_model_name = f"{self.model_name}--{date.today()}.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{output_dir}{tmp_model_name}", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
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
    permanet_model_name = f"{iou}--{self.num_to_use}--{tmp_model_name}"
    hist_csv_file = f'{output_dir}{permanet_model_name}-history.csv'
    os.system(f"mv {output_dir}{tmp_model_name} \
          {output_dir}{permanet_model_name}")
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

  def create_model(self,model_name):
    if model_name == 'custom':
      return self.custom_unet()
    else:
      return self.pretrained_model(model_name)

  def data_gen(self, num_to_use):
    data_paths = pathlib.Path(data_dir)
    input_img_paths = list(data_paths.glob("*/raw/*"))
    input_img_paths = sorted(input_img_paths, key=os.path.basename)
    target_img_paths = list(data_paths.glob("*/label/*"))
    target_img_paths = sorted(target_img_paths, key=os.path.basename)
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    input_img_paths = input_img_paths[:num_to_use]
    target_img_paths = target_img_paths[:num_to_use]
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
    train_gen = particles(batch_size, img_size, train_input_img_paths, train_target_img_paths,fold=1)
    val_gen = particles(batch_size, img_size, val_input_img_paths, val_target_img_paths,fold=1)
    test_gen = particles(batch_size, img_size, test_input_img_paths, test_target_img_paths, fold=1)
    return train_gen, val_gen, test_gen


  def custom_unet(self):
    inputs = tf.keras.Input(shape=img_size + (3,))
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", 
            metrics=[UpdatedMeanIoU(num_classes=num_classes)])
    return model
  
  def pretrained_model(self,model_name):
    class_method = getattr(tf.keras.applications, model_name)
    base_model = class_method(input_shape=[256, 256, 3], include_top=False, weights='imagenet')
    shape_128 = []
    shape_64 = []
    shape_32 = []
    shape_16 = []
    shape_8 = []
    for layer in base_model.layers:
        if layer.__class__.__name__ == 'Activation':
            if layer.input_shape[1:3] == (128,128):
                shape_128.append(layer.get_config()['name'])
            elif layer.input_shape[1:3] == (64,64):
                shape_64.append(layer.get_config()['name'])
            elif layer.input_shape[1:3] == (32,32):
                shape_32.append(layer.get_config()['name'])
            elif layer.input_shape[1:3] == (16,16):
                shape_16.append(layer.get_config()['name'])
            elif layer.input_shape[1:3] == (8,8):
                shape_8.append(layer.get_config()['name'])
    layer_names = [
                    shape_128[-1], # size 128*128
                    shape_64[-1],  # size 64*64
                    shape_32[-1],  # size 32*32
                    shape_16[-1],  # size 16*16
                    shape_8[-1]        # size 8*8
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names] 
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = True
    model = self.unet_model(output_channels=3, down_stack=down_stack)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[UpdatedMeanIoU(num_classes=num_classes)])
    return model
    
  def unet_model(self,output_channels:int, down_stack):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    up_stack = [
        pix2pix.upsample(1024, 3),  # 4x4 -> 8x8
        pix2pix.upsample(512, 3),  # 8x8 -> 16x16
        pix2pix.upsample(256, 3),  # 16x16 -> 32x32
        pix2pix.upsample(128, 3),   # 32x32 -> 64x64
    ]
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
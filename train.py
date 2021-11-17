# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import random
import numpy as np
import tensorflow as tf
import pathlib
from scipy.ndimage import gaussian_filter
import cv2
import pandas as pd
from datetime import date
from particle import *
mirroted_strategy = tf.distribute.MirroredStrategy()

lowpass = False
data_dir = 'data_for_training'
img_size = (256, 256)
num_classes = 3
batch_size = 32
data_aug_fold = 4

data_paths = pathlib.Path(data_dir)

input_img_paths = list(data_paths.glob("*/raw/*"))

input_img_paths = sorted(input_img_paths, key=os.path.basename)

target_img_paths = list(data_paths.glob("*/label/*"))
target_img_paths = sorted(target_img_paths, key=os.path.basename)
print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:5], target_img_paths[:5]):
    print(input_path, "|", target_path)

    
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
from tensorflow.keras import layers


def get_model(img_size, num_classes):
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
    return model


# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

# Build model
with mirroted_strategy.scope():
    model = get_model(img_size, num_classes)
    model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])
# model = get_model(img_size, num_classes)
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])
#model = keras.models.load_model("oxford_segmentation.h5")
model.summary()


# %%

# Split our img paths into a training and a validation set
val_samples = int(len(input_img_paths)//5)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]


if lowpass:
    train_gen = lp_particles(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = lp_particles(batch_size, img_size, val_input_img_paths, val_target_img_paths)
else:
    train_gen = particles(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = particles(batch_size, img_size, val_input_img_paths, val_target_img_paths)


# %%
#opt = keras.optimizers.Adam(learning_rate=5e-4)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f"models/particle_segmentation_lp-{date.today()}.h5", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
history = model.fit(train_gen, epochs=epochs,  callbacks=callbacks, validation_data=val_gen,)
hist_df = pd.DataFrame(history.history)

hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


# %%




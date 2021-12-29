# %%
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow_hub as hub
from datetime import date
import sys
class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, image_size, shuffle=True):
        self.df = df.copy() 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.img_size = image_size
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    def __getitem__(self, index):
        i = index*self.batch_size
        batch_df = self.df[i : i + self.batch_size].reset_index(drop=True)
        X = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        y = np.zeros((self.batch_size,), dtype="float64")
        for j in range(len(batch_df)):
            path = self.df.iloc[j]['name']
            path = f'preds/{path}'
            image = np.load(path)
            image = cv2.resize(image, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            image = np.stack((image,)*3, axis=-1)
            X[j] = image
            y[j] = self.df.iloc[j]['iou']
        return X,y
    def __len__(self):
        return self.n//self.batch_size
df = pd.read_csv('preds/preds_iou.csv', index_col=0)
#df = df[:1000]
train_split = len(df)//5*4
train_df = df[:train_split].reset_index(drop=True)
val_df = df[train_split:].reset_index(drop=True)
def pred_single(val_df,index, model, i):
    i += 1
    test_image_name = val_df.iloc[index]['name']
    test_image_path = f'preds/{test_image_name}'
    test_image_raw = np.load(test_image_path)
    test_image = cv2.resize(test_image_raw, dsize=IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    test_image = np.stack((test_image,)*3, axis=-1)
    test_image = np.expand_dims(test_image,axis=0)
    pred_label = model.predict(test_image)
    true_label = val_df.iloc[index]['iou']
    #plt.figure()
    plt.subplot(3,3,i)
    plt.imshow(test_image_raw)
    plt.title(f'True:{true_label} \n Pred:{pred_label[0][0]}')
    plt.axis('off')

# %%
model_name = sys.argv[1] # @param ['efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l', 'efficientnetv2-s-21k', 'efficientnetv2-m-21k', 'efficientnetv2-l-21k', 'efficientnetv2-xl-21k', 'efficientnetv2-b0-21k', 'efficientnetv2-b1-21k', 'efficientnetv2-b2-21k', 'efficientnetv2-b3-21k', 'efficientnetv2-s-21k-ft1k', 'efficientnetv2-m-21k-ft1k', 'efficientnetv2-l-21k-ft1k', 'efficientnetv2-xl-21k-ft1k', 'efficientnetv2-b0-21k-ft1k', 'efficientnetv2-b1-21k-ft1k', 'efficientnetv2-b2-21k-ft1k', 'efficientnetv2-b3-21k-ft1k', 'efficientnetv2-b0', 'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'bit_s-r50x1', 'inception_v3', 'inception_resnet_v2', 'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'nasnet_large', 'nasnet_mobile', 'pnasnet_large', 'mobilenet_v2_100_224', 'mobilenet_v2_130_224', 'mobilenet_v2_140_224', 'mobilenet_v3_small_100_224', 'mobilenet_v3_small_075_224', 'mobilenet_v3_large_100_224', 'mobilenet_v3_large_075_224']

model_handle_map = {
  "efficientnetv2-s": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
  "efficientnetv2-m": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_m/feature_vector/2",
  "efficientnetv2-l": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2",
  "efficientnetv2-s-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_s/feature_vector/2",
  "efficientnetv2-m-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_m/feature_vector/2",
  "efficientnetv2-l-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_l/feature_vector/2",
  "efficientnetv2-xl-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2",
  "efficientnetv2-b0-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b0/feature_vector/2",
  "efficientnetv2-b1-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b1/feature_vector/2",
  "efficientnetv2-b2-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2",
  "efficientnetv2-b3-21k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b3/feature_vector/2",
  "efficientnetv2-s-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
  "efficientnetv2-m-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
  "efficientnetv2-l-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
  "efficientnetv2-xl-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
  "efficientnetv2-b0-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b0/feature_vector/2",
  "efficientnetv2-b1-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b1/feature_vector/2",
  "efficientnetv2-b2-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2",
  "efficientnetv2-b3-21k-ft1k": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b3/feature_vector/2",
  "efficientnetv2-b0": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
  "efficientnetv2-b1": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b1/feature_vector/2",
  "efficientnetv2-b2": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b2/feature_vector/2",
  "efficientnetv2-b3": "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
  "efficientnet_b0": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b0/feature-vector/1",
  "efficientnet_b1": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b1/feature-vector/1",
  "efficientnet_b2": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b2/feature-vector/1",
  "efficientnet_b3": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b3/feature-vector/1",
  "efficientnet_b4": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b4/feature-vector/1",
  "efficientnet_b5": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b5/feature-vector/1",
  "efficientnet_b6": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b6/feature-vector/1",
  "efficientnet_b7": "https://hub.tensorflow.google.cn/tensorflow/efficientnet/b7/feature-vector/1",
  "bit_s-r50x1": "https://hub.tensorflow.google.cn/google/bit/s-r50x1/1",
  "inception_v3": "https://hub.tensorflow.google.cn/google/imagenet/inception_v3/feature-vector/4",
  "inception_resnet_v2": "https://hub.tensorflow.google.cn/google/imagenet/inception_resnet_v2/feature-vector/4",
  "resnet_v1_50": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v1_50/feature-vector/4",
  "resnet_v1_101": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v1_101/feature-vector/4",
  "resnet_v1_152": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v1_152/feature-vector/4",
  "resnet_v2_50": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/feature-vector/4",
  "resnet_v2_101": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_101/feature-vector/4",
  "resnet_v2_152": "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_152/feature-vector/4",
  "nasnet_large": "https://hub.tensorflow.google.cn/google/imagenet/nasnet_large/feature_vector/4",
  "nasnet_mobile": "https://hub.tensorflow.google.cn/google/imagenet/nasnet_mobile/feature_vector/4",
  "pnasnet_large": "https://hub.tensorflow.google.cn/google/imagenet/pnasnet_large/feature_vector/4",
  "mobilenet_v2_100_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
  "mobilenet_v2_130_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_130_224/feature_vector/4",
  "mobilenet_v2_140_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
  "mobilenet_v3_small_100_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5",
  "mobilenet_v3_small_075_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_small_075_224/feature_vector/5",
  "mobilenet_v3_large_100_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5",
  "mobilenet_v3_large_075_224": "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5",
}

model_image_size_map = {
  "efficientnetv2-s": 384,
  "efficientnetv2-m": 480,
  "efficientnetv2-l": 480,
  "efficientnetv2-b0": 224,
  "efficientnetv2-b1": 240,
  "efficientnetv2-b2": 260,
  "efficientnetv2-b3": 300,
  "efficientnetv2-s-21k": 384,
  "efficientnetv2-m-21k": 480,
  "efficientnetv2-l-21k": 480,
  "efficientnetv2-xl-21k": 512,
  "efficientnetv2-b0-21k": 224,
  "efficientnetv2-b1-21k": 240,
  "efficientnetv2-b2-21k": 260,
  "efficientnetv2-b3-21k": 300,
  "efficientnetv2-s-21k-ft1k": 384,
  "efficientnetv2-m-21k-ft1k": 480,
  "efficientnetv2-l-21k-ft1k": 480,
  "efficientnetv2-xl-21k-ft1k": 512,
  "efficientnetv2-b0-21k-ft1k": 224,
  "efficientnetv2-b1-21k-ft1k": 240,
  "efficientnetv2-b2-21k-ft1k": 260,
  "efficientnetv2-b3-21k-ft1k": 300, 
  "efficientnet_b0": 224,
  "efficientnet_b1": 240,
  "efficientnet_b2": 260,
  "efficientnet_b3": 300,
  "efficientnet_b4": 380,
  "efficientnet_b5": 456,
  "efficientnet_b6": 528,
  "efficientnet_b7": 600,
  "inception_v3": 299,
  "inception_resnet_v2": 299,
  "nasnet_large": 331,
  "pnasnet_large": 331,
}

model_handle = model_handle_map.get(model_name)
pixels = model_image_size_map.get(model_name, 224)

print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 32

# %%
do_fine_tuning = False
print("Building model with", model_handle)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.3, 
                                                                width_factor=0.3,
                                                                fill_mode='constant',
                                                                fill_value=1),
    hub.KerasLayer(model_handle, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(1)
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

# %%
traingen = CustomDataGen(train_df,batch_size=BATCH_SIZE,image_size= IMAGE_SIZE)
valgen = CustomDataGen(val_df, batch_size=BATCH_SIZE, image_size = IMAGE_SIZE)
base_learning_rate = 0.001
base_model_name = f'pred_rating_{model_name}_{date.today()}'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f"models/{base_model_name}.h5", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='mean_absolute_error')
steps_per_epoch = len(train_df) // BATCH_SIZE
validation_steps = len(val_df) // BATCH_SIZE
history = model.fit(
    traingen,
    callbacks=callbacks,
    epochs=30, steps_per_epoch=steps_per_epoch,
    validation_data=valgen,
    validation_steps=validation_steps)

# %%
# data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.RandomFlip('horizontal'),
#   tf.keras.layers.RandomRotation(0.2),
# ])

# preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
# IMG_SHAPE = IMAGE_SIZE + (3,)
# base_model = tf.keras.applications.resnet_v2.ResNet152V2(input_shape=IMG_SHAPE, include_top = False, weights = 'imagenet')
# base_model.trainable = False
# image_batch, label_batch = next(iter(traingen))
# feature_batch = base_model(image_batch)
# print(feature_batch.shape)
# global_average_layer =tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)
# prediction_layer = tf.keras.layers.Dense(1)
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)

# inputs = tf.keras.Input(shape=(224, 224, 3))

# #x = data_augmentation(inputs)
# x = inputs
# x = preprocess_input(x)
# x = base_model(x, training=False)
# x = global_average_layer(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# outputs = prediction_layer(x)

# model = tf.keras.Model(inputs, outputs)

# # %%
# base_learning_rate = 0.001
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#               loss='mean_absolute_error')
# callback = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
#     tf.keras.callbacks.ModelCheckpoint(f"models/pred_rating_model.h5", save_best_only=True)
# ]
# # loss0= model.evaluate(valgen)
# # val_pred0 = model.predict(valgen)
# # plt.figure(figsize=(10,11))
# # for i in range(9):
# #     idx = i
# #     #idx = random.randint(0,len(val_df))
# #     pred_single(val_df,idx,model,i)
# # %%
# history = model.fit(traingen,
#                     epochs=20,
#                     validation_data=valgen, callbacks=[callback])
# # %%
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# # plt.figure(figsize=(8, 8))

# # plt.subplot(1, 1, 1)
# # plt.plot(loss, label='Training Loss')
# # plt.plot(val_loss, label='Validation Loss')
# # plt.legend(loc='upper right')
# # plt.ylabel('Cross Entropy')
# # #plt.ylim([0,1.0])
# # plt.title('Training and Validation Loss')
# # plt.xlabel('epoch')
# # plt.show()
# # %%

model = tf.keras.models.load_model(f"models/{base_model_name}.h5",custom_objects={"KerasLayer": hub.KerasLayer})
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss="mean_absolute_error")
loss_now= model.evaluate(valgen)
loss_now = "{:.4f}".format(loss_now)
hist_df = pd.DataFrame(history.history)
os.system(f"mv models/{base_model_name}.h5 models/{loss_now}--{base_model_name}.h5")
hist_csv_file = f'models/{loss_now}--{base_model_name}.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
# plt.figure(figsize=(10,11))
# for i in range(9):
#     idx = i
#     idx = random.randint(0,len(val_df))
#     pred_single(val_df,idx,model,i)
# %%



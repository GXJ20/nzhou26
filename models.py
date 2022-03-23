import tensorflow as tf
from tensorflow.keras import layers
from helper import UpdatedMeanIoU
from tensorflow_examples.models.pix2pix import pix2pix
img_size = (256, 256)
num_classes = 3
# this script create different types of unet models

# this model was inspired by official keras tutorial
def custom_unet():
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
        previous_block_activation = x 
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

# this function combines established archtecture as downsampling block
def pretrained_model(model_name, fine_tune_at):
    class_method = getattr(tf.keras.applications, model_name)
    base_model = class_method(input_shape=[256, 256, 3], include_top=False, weights='imagenet')
    shape_128 = []
    shape_64 = []
    shape_32 = []
    shape_16 = []
    shape_8 = []
    # get activate layers, or it could be intepreted as the layers reduce size
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
    # construct the downsampling block
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names] 
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = True
    print(len(down_stack.layers))
    # set number of layers that are trainable
    for layer in down_stack.layers[:fine_tune_at]:
        layer.trainable = False
    
    # it calls the upsampling block and return a whole model
    model = unet_model(output_channels=3, down_stack=down_stack)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[UpdatedMeanIoU(num_classes=num_classes)])
    return model
    
# this function constrcut downsampling block, which is reusable by all upsampling block
def unet_model(output_channels, down_stack):
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
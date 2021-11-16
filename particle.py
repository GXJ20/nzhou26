import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import gaussian_filter
class particles(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        shift = 0.3
        #raw_shift = tf.image.random_crop(raw_shift, size = seed=seed)
        h, w = self.img_size
        tx = np.random.uniform(-shift, shift)*h
        ty = np.random.uniform(-shift, shift)*w
        #print(tx,ty)
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            #img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.expand_dims(img,2)
            #img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
            img -= 1
            y[j] = img
        #print(x.shape,y.shape)
        return x, y

class lp_particles(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        shift = 0.3
        #raw_shift = tf.image.random_crop(raw_shift, size = seed=seed)
        h, w = self.img_size
        tx = np.random.uniform(-shift, shift)*h
        ty = np.random.uniform(-shift, shift)*w
        #print(tx,ty)
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            img = gaussian_filter(img, sigma=5)
            #img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.expand_dims(img,2)
            #img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
            img -= 1
            y[j] = img
        #print(x.shape,y.shape)
        return x, y

class inference_particles(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            #img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
            x[j] = img
        #print(x.shape,y.shape)
        return x
class lp_inference_particles(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            #img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
            img = gaussian_filter(img, sigma=5)
            x[j] = img
        #print(x.shape,y.shape)
        return x

import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import gaussian_filter
tf.get_logger().setLevel('ERROR')

class particles(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, fold=1):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.fold = fold
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
    
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        if self.fold == 1:
            x,y = self.__transform(batch_input_img_paths, batch_target_img_paths, 0.3)
        elif self.fold == 0:
            x,y = self.__transform(batch_input_img_paths, batch_target_img_paths, 0)
        else:
            x,y = self.__augmentation(batch_input_img_paths, batch_target_img_paths, self.fold)
        return x, y
    def __transform(self, batch_input_img_paths, batch_target_img_paths, shift):
        h, w = self.img_size
        tx = np.random.uniform(-shift, shift)*h
        ty = np.random.uniform(-shift, shift)*w
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty, channel_axis=2, fill_mode='nearest')
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.load(path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.expand_dims(img,2)
            img = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
            img -= 1
            y[j] = img
        return x,y
    def __augmentation(self, batch_input_img_paths, batch_target_img_paths, fold):
        shift = 0.3
        h, w = self.img_size
        x = np.zeros((self.batch_size*fold,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size*fold,) + self.img_size + (1,), dtype="uint8")
        for j, paths in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
            img_path,target_path = paths
            img = np.load(img_path)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            x[j] = img
            tgt = np.load(target_path)
            tgt = cv2.resize(tgt, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            tgt = np.expand_dims(tgt,2)
            tgt -= 1
            y[j] = tgt
            for i in range(1,fold):
                tx = np.random.uniform(-shift, shift)*h
                ty = np.random.uniform(-shift, shift)*w
                img_aug = tf.keras.preprocessing.image.apply_affine_transform(img, tx,ty , channel_axis=2, fill_mode='nearest')
                tgt_aug = tf.keras.preprocessing.image.apply_affine_transform(tgt, tx,ty , channel_axis=2, fill_mode='nearest')
                x[j + self.batch_size*i] = img_aug
                y[j + self.batch_size*i] = tgt_aug
        np.random.seed(42)
        np.random.shuffle(x)
        np.random.seed(42)
        np.random.shuffle(y)
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
            img = np.load(path,allow_pickle=True)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            x[j] = img
        return x
class rating_particles(tf.keras.utils.Sequence):
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
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        for j, path in enumerate(batch_input_img_paths):
            img = np.load(path,allow_pickle=True)
            img = cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            x[j] = img
        return x



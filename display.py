# %%
import pathlib
from scipy.ndimage.interpolation import rotate
import starfile
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import tensorflow_hub as hub
import pandas as pd
seg_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/'
rating_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/rating/'
img_size = (256,256)

class plot_models():
    def __init__(self, model_dir, pattern):
        model_paths = list(pathlib.Path(model_dir).glob(pattern))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
        model_metrics = [float(item.name.split('--')[0]) for item in model_paths]
        model_names =  ['@'.join(item.name.split('--')[2:4]).split(".")[0] for item in model_paths]
        
        self.display(model_metrics, model_names)
    def display(self, metrics, names):
        short_names = [f'{name[:10]}...' for name in names]
        plt.figure(figsize=(5,5))
        plt.plot(metrics)
        plt.ylim(0,100)
        plt.xticks(ticks=range(0,len(names)),labels=short_names, rotation=85)
        plt.show()
        for i in range(0,len(names)):
            print(f'{names[i]}: {metrics[i]}')
class display_infer_particles():
    def __init__(self, star_file, raw_dir,num_to_display):
        raw_imgs = []
        lp_imgs = []
        imgs = []
        for i in range(num_to_display):
            raw_img = self.raw_from_star(i, star_file, raw_dir)
            raw_imgs.append(raw_img)
            lp_imgs.append(gaussian_filter(raw_img, sigma=5))
            img = cv2.resize(raw_img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
            img = np.stack((img,)*3, axis=-1)
            imgs.append(img)
        width = round(np.sqrt(num_to_display))
        self.display_batch(raw_imgs, width)
        self.display_batch(lp_imgs, width)
        pred_batch = np.asarray(imgs)
        pred = self.segment(pred_batch)
        self.display_batch(pred, width)
        ratings = self.rate(pred)
        self.display_batch(pred,width, titles=ratings)

    def display_batch(self,batch, width,titles=[]):
        plt.figure(figsize=(5,5))
        for i in range(1, width*width +1):
            plt.subplot(width, width, i)
            plt.imshow(batch[i-1], cmap='gray')
            if titles:
                plt.title(titles[i-1])
            plt.axis('off')
        plt.show()

    def raw_from_star(self, idx, star, raw_dir):
        metadata = starfile.read(star)
        raw_image_file_name= metadata.iloc[idx]['rlnImageName'].split('/')[-1]
        particle_idx = metadata.iloc[idx]['rlnImageName'].split('@')[0]
        raw_image_path = f"{raw_dir}{raw_image_file_name}"
        with mrcfile.open(raw_image_path) as mrc:
            raw_img = mrc.data[int(particle_idx)-1]
        return raw_img

    def segment(self,batch):
        model_paths =  list(pathlib.Path(seg_model_dir).glob('*.h5'))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
        seg_model = model_paths[0]
        print(f'Using segmentation model: {seg_model.name}')
        model = tf.keras.models.load_model(seg_model, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
        model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])
        pred_result = model.predict(batch)
        pred = np.argmax(pred_result,axis=-1)
        return pred

    def rate(self,batch):
        model_paths =  list(pathlib.Path(rating_model_dir).glob('*.h5'))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0])    
        rating_model_name = model_paths[0]
        print(f'Using rating model: {rating_model_name.name}')
        rating_model = tf.keras.models.load_model(rating_model_name, custom_objects={"KerasLayer": hub.KerasLayer})
        rating_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.001),
                    loss='mean_absolute_error')
        resize_batch = []
        resize_size = rating_model.layers[0].input_shape[1:3]
        for seg in batch:
            seg = cv2.resize(seg, dsize=resize_size, interpolation=cv2.INTER_NEAREST)
            seg = np.stack((seg,)*3, axis=-1)
            resize_batch.append(seg)
        resize_batch = np.asarray(resize_batch)
        ratings = rating_model.predict(resize_batch).tolist()
        return ratings
class check_seg_history():
    def __init__(self, csv_file):
        history = pd.read_csv(csv_file)
        plt.plot(history.iloc[:,1], label='val_loss')
        plt.plot(history.iloc[:,2], label = 'loss')
        plt.plot(history.iloc[:,3], label='val_mean_IOU')
        plt.plot(history.iloc[:,4], label = 'mean_IOU')
        plt.legend()
        plt.show()

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

star_file = '/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job011/particles.star'
raw_dir = '/storage_data/zhou_Ningkun/relionProject/particleSeg_ranking_rhs/Extract/job011/goodmrc_auto/'

if __name__ =='__main__':
    # display_test_particles
    #display_infer_particles(star_file, raw_dir, 16)
    #plot_models(seg_model_dir, '*290000*.csv')
    check_seg_history('/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/66.32--290000--DenseNet169--2021-12-28.h5-history.csv')
# %%

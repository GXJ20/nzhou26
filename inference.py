# %%
import starfile
import mrcfile
import os
import numpy as np
import tensorflow as tf
import pathlib
import pandas as pd
import cv2
from train import UpdatedMeanIoU
# %%
seg_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/'
rating_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/rating/'
img_size = (256,256)
class Inference_star():
    def __init__(self, star_file, raw_dir, batch_size=128, seg_model_key='*.h5') -> None:
        self.raw_dir = raw_dir
        self.star_file = star_file
        self.metadata = starfile.read(star_file)
        self.batch_size = batch_size
        model_paths =  list(pathlib.Path(seg_model_dir).glob(seg_model_key))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
        self.seg_model = model_paths[0]
        self.model_IOU = self.seg_model.name.split('--')[0]
        #self.seg_model = 'models/67.58--290000--particle_segmentation-2021-12-15.h5'

    def infer_batch(self, batch_number):
        start = batch_number * self.batch_size
        end = start + self.batch_size
        names = []
        imgs = []
        for i in range(start, end):
            try:
                raw_img, imageName = raw_from_star(i, self.metadata, self.raw_dir)
                img = cv2.resize(raw_img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
                img = np.stack((img,)*3, axis=-1)
                names.append(imageName)
                imgs.append(img)
            except IndexError:
                break
        pred_batch = np.asarray(imgs)
        pred = segment(pred_batch, seg_model_name=self.seg_model)
        ratings = ets(pred)
        segment_ets = pd.DataFrame({
            'ets_ratio':ratings,
            'name':names})
        return segment_ets
    def drop(self, ets_rank, drop_ratio):
        ets_rank = ets_rank.sort_values('ets_ratio')
        ets_rank = ets_rank.reset_index(drop=True)
        num_to_drop = int(len(ets_rank)*float(drop_ratio))
        ImageNames = ets_rank['name'].to_list()[:num_to_drop]
        metadata_dropped = self.metadata[~self.metadata['rlnImageName'].isin(ImageNames)]
        print('Cutoff: ')
        print(ets_rank.iloc[num_to_drop]['ets_ratio'])
        self.model_MAE = 'ets'
        new_star = f'{os.path.dirname(self.star_file)}/IOU{self.model_IOU}--dropped{drop_ratio}--{os.path.basename(self.star_file)}'
        sorted_backup_csv = f'{os.path.dirname(self.star_file)}/IOU{self.model_IOU}--dropped{drop_ratio}--backup.csv'
        ets_rank.to_csv(sorted_backup_csv)
        starfile.write(metadata_dropped, new_star, overwrite=True)
        print(f'Done and saved cleaned metadata in {new_star}')
def raw_from_star(idx, metadata, raw_dir):
    raw_image_file_name= metadata.iloc[idx]['rlnImageName'].split('/')[-1]
    particle_idx = metadata.iloc[idx]['rlnImageName'].split('@')[0]
    raw_image_path = f"{raw_dir}{raw_image_file_name}"
    with mrcfile.open(raw_image_path) as mrc:
        raw_img = mrc.data[int(particle_idx)-1]
    return raw_img, metadata.iloc[idx]['rlnImageName']

def segment(batch, seg_model_name):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = tf.keras.models.load_model(seg_model_name, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
        model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])
    pred_result = model.predict(batch, use_multiprocessing=True, workers=2)
    
    pred = np.argmax(pred_result,axis=-1)
    return pred
    
def ets(batch):
    ets_ratios = []
    for seg in batch:
        unique, counts = np.unique(seg, return_counts=True)
        my_dict = dict(zip(unique, counts))
        try:
            ets = my_dict[2]/my_dict[0]
        except KeyError:
            ets = 0
        ets_ratios.append(ets)
    ets_ratios = ["{:.4f}".format(num)  for num in ets_ratios]
    return ets_ratios

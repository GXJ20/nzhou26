# %%
from matplotlib.pyplot import axis
import starfile
import mrcfile
import os
import multiprocessing
import numpy as np
import tensorflow as tf
import pathlib
from particle import inference_particles, rating_particles
import pandas as pd
import tensorflow_hub as hub
import random
import cv2
# %%
# read data
output_folder = '/ssd/particleSeg/'
seg_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/'
rating_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/rating/'
process_number = 16
mirroted_strategy = tf.distribute.MirroredStrategy()
ets_ratios = []
img_size = (256,256)

class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, start_idx, end_idx, task, segmented_npy_paths=None,raw_data_dir=None, input_img_paths=None, df=None, pred_result=None):
        multiprocessing.Process.__init__(self)
        self.processID = processID
        self.name = name
        self.df = df
        self.input_img_paths = input_img_paths
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.task = task
        self.pred_result = pred_result
        self.raw_data_dir = raw_data_dir
        self.segmented_npy_paths = segmented_npy_paths
    def run(self):
        if self.task == 'gen_npy':
            generate_npy(self.name, self.df, self.raw_data_dir, self.start_idx, self.end_idx)
        elif self.task == 'batch_pred':
            batch_pred(self.name, self.pred_result, self.input_img_paths, self.start_idx, self.end_idx)
        elif self.task == 'edge_to_signal':
            edge_to_signal(self.name, self.segmented_npy_paths, self.start_idx, self.end_idx)

def generate_npy(processName, star, raw_data_dir,idx, end_idx):
    while idx < end_idx:
        item = star['rlnImageName'][idx]
        try:
            particle_idx = int(item.split("@")[0])
            particle_path = item.split("@")[1]
            particle_file = particle_path.split("/")[-1]
            if not os.path.exists(f"{output_folder}{particle_idx}@{particle_file}.npy"):
                with mrcfile.open(f"{raw_data_dir}{particle_file}") as mrc:
                    particle_raw = mrc.data[particle_idx-1]
                    np.save(f"{output_folder}{particle_idx}@{particle_file}",particle_raw)
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end_idx)
            print(f"Converting mrcs to npy, progress: {percent}")
        idx += 1

def batch_pred(processName, pred_result, input_img_paths, idx, end):
    for idx in range(idx, end):
        try:
            pred_idx = idx % process_number
            pred = np.argmax(pred_result[pred_idx],axis=-1)
            if not os.path.exists(f'{output_folder}/pred/{input_img_paths[idx].name}'):        
                np.save(f'{output_folder}/pred/{input_img_paths[idx].name}', pred)
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        # if processName == "Process-0":
        #     percent = "{:.2%}".format(idx/end)
        #     print(f"Running particle segmentation, progress: {percent}")

def edge_to_signal(processName, segmented_npy_paths, idx, end):
    ets_ratios = []
    names = []
    for idx in range(idx, end):
        try:
            seg = np.load(segmented_npy_paths[idx])
            unique, counts = np.unique(seg, return_counts=True)
            my_dict = dict(zip(unique, counts))
            ets = my_dict[2]/my_dict[0]
            ets_ratios.append(ets)
            names.append(segmented_npy_paths[idx].name)
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end)
            print(f"Computing edge to signal ratio, progress: {percent}")
    segment_ets = pd.DataFrame({
            'ets_ratios':ets_ratios,
            'name':names})
    pred_csv = f'{output_folder}pred/{processName}-seg_ets.csv'
    segment_ets.to_csv(pred_csv)

class Inference_star():
    def __init__ (self, star_file, raw_data_dir, ratio, continues=False):
        self.star_file= star_file
        self.metadata = starfile.read(star_file)
        self.raw_data_dir = raw_data_dir
        self.ratio = float(ratio)
        if not continues:
            os.system(f'rm -rf {output_folder}')
        os.system(f'mkdir -p {output_folder}')
        # self.npy_gen_mpi()
        # total_img_paths = list(pathlib.Path(output_folder).glob("*.npy"))
        # num_of_batches = len(total_img_paths) // 10240 +1
        self.pred_path = f'{output_folder}/pred/'
        self.pred_csv = f'{self.pred_path}seg_ets.csv'
        # os.system(f'mkdir -p {self.pred_path}')
        # for i in range(num_of_batches):
        #     print(f'running pred batch {i} out of {num_of_batches} batches')
        #     start = i* 10240
        #     if i == num_of_batches -1 :
        #         end = len(total_img_paths)
        #     else:
        #         end = start + 10240
        #     self.seg_mpi(total_img_paths[start:end])
        # #self.rating()
        #self.rating_new()
        self.ranking()
    def npy_gen_mpi(self):
        particles_to_be_done = len(self.metadata)
        particles_for_each_process = particles_to_be_done // process_number
        processes = []
        for i in range(process_number):
            start_idx = i*particles_for_each_process
            end_idx = start_idx+particles_for_each_process
            if i == process_number -1 :
                end_idx = particles_to_be_done
            processes.append( myProcess(processID=i, name=f"Process-{i}", 
                            df=self.metadata, raw_data_dir=self.raw_data_dir,
                            start_idx=start_idx, end_idx=end_idx, task='gen_npy'))
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    
    def seg_mpi(self, input_img_paths):
        model_paths =  list(pathlib.Path(seg_model_dir).glob('67.58*.h5'))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
        seg_model = model_paths[0]
        print(f'Using segmentation model: {seg_model.name}')
        self.model_IOU = seg_model.name.split('--')[0]
        tf.keras.backend.clear_session()
        with mirroted_strategy.scope():
            model = tf.keras.models.load_model(seg_model, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
            model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])
        
        
        num_of_particles = len(input_img_paths)
        particles_for_each_process = num_of_particles // process_number
        processes = []
        for i in range(process_number):
            start_idx = i*particles_for_each_process
            end_idx = start_idx+particles_for_each_process
            if i == process_number -1 :
                end_idx = num_of_particles
            pred_gen = inference_particles(batch_size=16, img_size= (256,256),input_img_paths=input_img_paths[start_idx:end_idx])
            pred_result = model.predict(pred_gen, verbose=1)#, use_multiprocessing=True, workers=4)
            processes.append( myProcess(processID=i,
                name= f"Process-{i}",
                start_idx= start_idx, 
                pred_result=pred_result,
                end_idx = end_idx, 
                input_img_paths=input_img_paths,
                task='batch_pred'))
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    def rating(self):
        model_paths =  list(pathlib.Path(rating_model_dir).glob('*.h5'))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0])    
        rating_model_name = model_paths[0]
        print(f'Using rating model: {rating_model_name.name}')
        self.model_MAE = rating_model_name.name.split('--')[0]
        rating_model = tf.keras.models.load_model(rating_model_name, custom_objects={"KerasLayer": hub.KerasLayer})
        rating_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.001),
                    loss='mean_absolute_error')
        names = []
        ious = []
        pred_result_paths = list(pathlib.Path(self.pred_path).glob("*.npy"))
        num_of_particles = len(pred_result_paths)
        particles_for_each_process = num_of_particles // process_number
        particles_for_each_process = particles_for_each_process//16 *16
        for i in range(process_number):
            start_idx = i*particles_for_each_process
            end_idx = start_idx+particles_for_each_process
            pred_gen = rating_particles(batch_size=16, img_size= (224,224),input_img_paths=pred_result_paths[start_idx:end_idx])
            iou_result = rating_model.predict(pred_gen, verbose=1, use_multiprocessing=True, workers=4)
            name = [item.name for item in pred_result_paths[start_idx:end_idx]]
            names.append(name)
            ious.append(iou_result)
        names = np.concatenate(names, axis=None)
        ious = np.concatenate(ious, axis=None)
        pred_iou = pd.DataFrame({
            'iou':ious,
            'name':names})
        self.pred_csv = f'{self.pred_path}{self.model_MAE}--preds_iou.csv'
        pred_iou.to_csv(self.pred_csv)
        
    
    def rating_new(self):
        pred_result_paths = list(pathlib.Path(self.pred_path).glob("Proc*.npy"))
        num_of_particles = len(pred_result_paths)
        particles_for_each_process = num_of_particles // process_number
        particles_for_each_process = particles_for_each_process//16 *16
        names = []
        processes = []
        for i in range(process_number):
            start_idx = i*particles_for_each_process
            end_idx = start_idx+particles_for_each_process
            processes.append( myProcess(processID=i,
                name= f"Process-{i}",
                start_idx= start_idx, 
                segmented_npy_paths=pred_result_paths,
                end_idx = end_idx, 
                task='edge_to_signal'))
            name = [item.name for item in pred_result_paths[start_idx:end_idx]]
            names.append(name)
        names = np.concatenate(names, axis=None)
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        csv_list = []
        for item in pathlib.Path(self.pred_path).glob('*.csv'):
            print(item)
            df = pd.read_csv(item, index_col=None)
            csv_list.append(df)
        total_ets = pd.concat(csv_list, axis=0, ignore_index=True)
        self.pred_csv = f'{self.pred_path}seg_ets.csv'
        total_ets.to_csv(self.pred_csv)
    
    def rating_new_single(self):
        pred_result_paths = list(pathlib.Path(self.pred_path).glob("*.npy"))
        ets_ratios = []
        for i in range(len(pred_result_paths)):
            names = [item.name for item in pred_result_paths]
            seg = np.load(pred_result_paths[i])
            unique, counts = np.unique(seg, return_counts=True)
            my_dict = dict(zip(unique, counts))
            ets = my_dict[2]/my_dict[0]
            ets_ratios.append(ets)
        print(ets_ratios)
        segment_ets = pd.DataFrame({
            'ets_ratios':ets_ratios,
            'name':names})
        
        segment_ets.to_csv(self.pred_csv)


    def ranking(self):
        iou_rank = pd.read_csv(self.pred_csv)
        #iou_rank = iou_rank.sort_values('iou')
        iou_rank = iou_rank.sort_values('ets_ratios')
        iou_rank = iou_rank.reset_index(drop=True)
        iou_rank.to_csv(f'{self.pred_path}sorted.csv')
        num_to_drop = int(len(iou_rank)*self.ratio)
        for i in range(num_to_drop):
            name = str(iou_rank.iloc[i]['name'])
            number_part = name.split('@')[0]
            name_part = name.split('@')[1][:-4]
            number_part = number_part.zfill(6)
            middle_part = self.metadata.iloc[0]['rlnImageName'].split('@')[1].rsplit('/',1)[0]
            ImageName = f'{number_part}@{middle_part}/{name_part}'
            self.metadata = self.metadata.drop(self.metadata[self.metadata['rlnImageName'] == ImageName].index)
            percent = "{:.2%}".format(i/num_to_drop)
            
            print(f"Dropping bad particles, progress: {percent}")
            if i == num_to_drop - 1:
                print(iou_rank.iloc[i]['ets_ratios'])
        self.model_MAE = 'ets'
        self.model_IOU = '67.58'
        starfile.write(self.metadata, f'{os.path.dirname(self.star_file)}/{self.model_IOU}--{self.model_MAE}{os.path.basename(self.star_file)}', overwrite=True)
        #os.system(f'rm -rf {output_folder}')
        print(f'Done and saved cleaned metadata in {os.path.dirname(self.star_file)}/{self.model_IOU}--{self.model_MAE}{os.path.basename(self.star_file)}')

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

class Inference_star_new():
    def __init__(self, star_file, raw_dir, batch_size=128, seg_model_key='*.h5') -> None:
        os.system(f'mkdir -p {output_folder}')
        self.raw_dir = raw_dir
        self.star_file = star_file
        self.metadata = starfile.read(star_file)
        self.batch_size = batch_size
        self.pred_csv = f'{output_folder}total.csv'
        model_paths =  list(pathlib.Path(seg_model_dir).glob(seg_model_key))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
        self.seg_model = model_paths[0]
        self.model_IOU = self.seg_model.name.split('--')[0]
    def infer_batch(self, batch_number):
        raw_imgs = []
        start = batch_number * self.batch_size
        end = start + self.batch_size
        names = []
        imgs = []
        for i in range(start, end):
            try:
                raw_img, imageName = self.raw_from_star(i, self.metadata, self.raw_dir)
                raw_imgs.append(raw_img)
                img = cv2.resize(raw_img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
                img = np.stack((img,)*3, axis=-1)
                names.append(imageName)
                imgs.append(img)
            except IndexError:
                break
        pred_batch = np.asarray(imgs)
        pred = self.segment(pred_batch, seg_model_name=self.seg_model)
        #ratings = self.rate(pred)
        ratings = self.ets(pred)
        segment_ets = pd.DataFrame({
            'ets_ratio':ratings,
            'name':names})
        segment_ets.to_csv(f'{output_folder}{batch_number}--ets.csv')
    def concat(self):
        csv_list = []
        for item in pathlib.Path(output_folder).glob('*ets.csv'):
            df = pd.read_csv(item, index_col=0)
            csv_list.append(df)
        total_ets = pd.concat(csv_list, axis=0, ignore_index=True)
        total_ets.to_csv(self.pred_csv)
    def drop(self, drop_ratio):
        ets_rank = pd.read_csv(self.pred_csv)
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
        #os.system(f'rm -rf {output_folder}')
        print(f'Done and saved cleaned metadata in {new_star}')
    def raw_from_star(self, idx, metadata, raw_dir):
        raw_image_file_name= metadata.iloc[idx]['rlnImageName'].split('/')[-1]
        particle_idx = metadata.iloc[idx]['rlnImageName'].split('@')[0]
        raw_image_path = f"{raw_dir}{raw_image_file_name}"
        with mrcfile.open(raw_image_path) as mrc:
            raw_img = mrc.data[int(particle_idx)-1]
        return raw_img, metadata.iloc[idx]['rlnImageName']

    def segment(self,batch, seg_model_name):
        
        model = tf.keras.models.load_model(seg_model_name, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
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
        ratings = ["{:.4f}".format(num[0])  for num in ratings]
        return ratings
        
    def ets(self, batch):
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

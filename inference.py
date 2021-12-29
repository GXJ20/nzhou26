# %%
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
# %%
# read data
output_folder = '/tmp/particleSeg/'
seg_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/segmentation/'
rating_model_dir = '/storage_data/zhou_Ningkun/workspace/data_particleSeg/models/rating/'
process_number = 16

class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, start_idx, end_idx, task, raw_data_dir=None, input_img_paths=None, df=None, pred_result=None):
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
    def run(self):
        if self.task == 'gen_npy':
            generate_npy(self.name, self.df, self.raw_data_dir, self.start_idx, self.end_idx)
        elif self.task == 'batch_pred':
            batch_pred(self.name, self.pred_result, self.input_img_paths, self.start_idx, self.end_idx)

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
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end)
            print(f"Running particle segmentation, progress: {percent}")
        idx += 1
class Inference_star():
    def __init__ (self, star_file, raw_data_dir):
        self.star_file= star_file
        self.metadata = starfile.read(star_file)
        self.raw_data_dir = raw_data_dir
    def e2e_infer(self):
        os.system(f'rm -rf {output_folder}')
        os.system(f'mkdir -p {output_folder}')
        self.npy_gen_mpi()
        self.seg_mpi()
        self.rating()
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
    
    def seg_mpi(self):
        model_paths =  list(pathlib.Path(seg_model_dir).glob('*.h5'))
        model_paths = sorted(model_paths, key=lambda model: model.name.split('--')[0], reverse=True)    
        seg_model = model_paths[0]
        print(f'Using segmentation model: {seg_model.name}')
        model = tf.keras.models.load_model(seg_model, custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
        model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy", metrics=[UpdatedMeanIoU(num_classes=3)])
        raw_npy_path = pathlib.Path(output_folder)
        input_img_paths = list(raw_npy_path.glob("*.npy"))
        self.pred_path = f'{output_folder}/pred/'
        os.system(f'mkdir -p {self.pred_path}')
        num_of_particles = len(input_img_paths)
        particles_for_each_process = num_of_particles // process_number
        processes = []
        for i in range(process_number):
            start_idx = i*particles_for_each_process
            end_idx = start_idx+particles_for_each_process
            if i == process_number -1 :
                end_idx = num_of_particles
            pred_gen = inference_particles(batch_size=16, img_size= (256,256),input_img_paths=input_img_paths[start_idx:end_idx])
            pred_result = model.predict(pred_gen, verbose=1, use_multiprocessing=False, workers=4)
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
    def ranking(self):
        iou_rank = pd.read_csv(self.pred_csv)
        iou_rank = iou_rank.sort_values('iou')
        iou_rank = iou_rank.reset_index(drop=True)
        num_to_drop = len(iou_rank) //3
        for i in range(num_to_drop):
            name = str(iou_rank.iloc[i]['name'])
            number_part = name.split('@')[0]
            name_part = name.split('@')[1][:-4]
            number_part = number_part.zfill(6)
            ImageName = f'{number_part}@Extract/job011/goodmrc_auto/{name_part}'
            self.metadata = self.metadata.drop(self.metadata[self.metadata['rlnImageName'] == ImageName].index)
            percent = "{:.2%}".format(i/num_to_drop)
            print(f"Dropping bad particles, progress: {percent}")
        starfile.write(self.metadata, f'{os.path.dirname(self.star_file)}/{self.model_MAE}{os.path.basename(self.star_file)}')
        os.system(f'rm -rf {output_folder}')
        print(f'Done and saved cleaned metadata in {os.path.dirname(self.star_file)}/{self.model_MAE}{os.path.basename(self.star_file)}')

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

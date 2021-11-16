# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import starfile
import mrcfile
import multiprocessing
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import pathlib

# edit these four parameters below
process_number = 17
project_id = "gpr158_3d"
raw_data_dir = f"gpr158_3d/Extract/job005/goodmrc_0123/"
model_path = "particle_segmentation.h5"


output_folder = f"pred_{project_id}"
# converted_png_path = f"{output_folder}png/"
# predicted_mask_path = f"{output_folder}mask/"
# altered_mrcs_path = f"{output_folder}mrcs/"
# os.system(f'mkdir -p {converted_png_path}')
# os.system(f'mkdir -p {predicted_mask_path}')
# os.system(f'mkdir -p {altered_mrcs_path}')

img_size = (256, 256)
num_classes = 3
batch_size = 256

class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, df, start_idx, end_idx, task):
        multiprocessing.Process.__init__(self)
        self.processID = processID
        self.name = name
        self.df = df
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.task = task
    def run(self):
        # print("start process: " + self.name)
        if self.task == "alter_mrcs":
            alter_particle(self.name, self.df, self.start_idx, self.end_idx)
        elif self.task == "gen_png":
            generate_png(self.name, self.df, self.start_idx, self.end_idx)
        # print("exit process: "+ self.name)

def generate_png(processName, star, idx, end_idx):
    while idx < end_idx:
        item = star['rlnImageName'][idx]
        try:
            particle_idx = int(item.split("@")[0])
            particle_path = item.split("@")[1]
            particle_file = particle_path.split("/")[-1]
            with mrcfile.open(f"{project_dir}{particle_path}") as mrc:
                particle_raw = mrc.data[particle_idx-1]
                plt.imsave(f"{converted_png_path}{particle_idx}@{particle_file}.png",particle_raw, cmap="gray")
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end_idx)
            print(f"Converting mrcs to png, progress: {percent}")
        idx += 1

def alter_particle(processName, groups, idx, end_idx):
    while idx < end_idx:
        single_mrc = groups[idx]
        try:
            particle_path = single_mrc.iloc[0]["rlnImageName"].split("@")[1]
            particle_file = particle_path.split("/")[-1]
            #print(f'converting {particle_file}')
            os.system(f'rsync {project_dir}{particle_path} {altered_mrcs_path}{particle_file}')
            with mrcfile.open(f"{altered_mrcs_path}{particle_file}", mode='r+') as mrc:
                for item in single_mrc['rlnImageName']:
                    particle_idx = int(item.split("@")[0])
                    mrc.data.setflags(write=1)
                    particle_raw = mrc.data[particle_idx-1]
                    mask_path = f"{predicted_mask_path}{particle_idx}@{particle_file}.png"
                    if os.path.isfile(mask_path):
                        mask = cv2.imread(mask_path)
                        mask = mask[:,:,0]
                        mask_resize = cv2.resize(mask, (particle_raw.shape), interpolation=cv2.INTER_NEAREST)
                        particle_raw[mask_resize == 127] = 0
                        mrc.data[particle_idx-1] = particle_raw
                mrc.set_data(mrc.data)
            #print(f'done {particle_file}')
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end_idx)
            print(f"Altering mrcs files, progress: {percent}")
        idx += 1

def gen_png_multi():
    star = starfile.open(star_path)
    particles_to_be_done = len(star)
    particles_for_each_process = particles_to_be_done // process_number
    processes = []
    for i in range(process_number):
        start_idx = i*particles_for_each_process
        end_idx = start_idx+particles_for_each_process
        if i == process_number -1 :
            end_idx = particles_to_be_done
        # print(i)
        # print(start_idx)
        # print(end_idx)
        processes.append( myProcess(i, f"Process-{i}", star, start_idx, end_idx, "gen_png"))

    for process in processes:
        process.start()
    for process in processes:
        process.join()

def alter_mrcs_multi():
    star = starfile.open(star_path)
    grouped = star.groupby('rlnMicrographName')
    groups = []
    for name, group in grouped:
        groups.append(group)
    mrcs_to_be_done = len(groups)
    mrcs_for_each_process = mrcs_to_be_done // process_number
    processes = []

    for i in range(process_number):
        start_idx = i*mrcs_for_each_process
        end_idx = start_idx+mrcs_for_each_process
        if i == process_number -1 :
            end_idx = mrcs_to_be_done
        # print(i)
        # print(start_idx)
        # print(end_idx)
        processes.append( myProcess(i, f"Process-{i}", groups, start_idx, end_idx, task="alter_mrcs"))

    for process in processes:
        process.start()
    for process in processes:
        process.join()

class t20s_particles(tf.keras.utils.Sequence):
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
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        return x

# %%
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

def pred_gen(start, end,data_img_paths):
    # print(start, end)
    model = tf.keras.models.load_model(f"{model_path}", custom_objects={"UpdatedMeanIoU": UpdatedMeanIoU})
    val_gen = t20s_particles(batch_size, img_size, data_img_paths[start:end])
    val_preds = model.predict(val_gen)
    # print(f"length is: {len(val_preds)}")
    return val_preds
def save_masks(i, val_preds, pred_batch, data_img_paths):
    idx_in_batch = i%pred_batch
    # print(f'true idx:{i}')
    # print(f'idx_in_batch: {idx_in_batch}')
    mask = np.argmax(val_preds[idx_in_batch], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    file_name = data_img_paths[i].split("/")[-1]
    plt.imsave(f"{predicted_mask_path}{file_name}",tf.keras.preprocessing.image.array_to_img(mask), cmap='gray')

# %%
def run_prediction():
    data_img_paths = sorted(
        [
            os.path.join(converted_png_path, fname)
            for fname in os.listdir(converted_png_path)
            if fname.endswith(".mrcs.png")
        ]
    )
    pred_batch = batch_size*10
    num_iters = len(data_img_paths)//pred_batch
    for i in range(num_iters):
        #print(f"running iteration {i} out of {num_iters}")
        start = i*pred_batch
        end = (i+1)*pred_batch
        val_preds = pred_gen(start, end, data_img_paths)
        for j in range(start, end):
            save_masks(j, val_preds, pred_batch, data_img_paths)
        percent = "{:.2%}".format(i/num_iters)
        print(f"Predicting masks, progress: {percent}")
def inference():
    data_img_paths = list(pathlib.Path(raw_data_dir).glob('*.mrcs'))
    for item in data_img_paths:
        print(item)
if __name__ == "__main__":
    inference()
    # gen_png_multi()
    # run_prediction()
    # alter_mrcs_multi()





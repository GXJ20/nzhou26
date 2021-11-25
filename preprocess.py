# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import mrcfile
import starfile
import os
import numpy as np
import multiprocessing
from scipy.ndimage import convolve
from datetime import datetime
import pandas as pd

train_data_path = 'data_for_training.json'

class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, meta_data, start_idx, end_idx):
        multiprocessing.Process.__init__(self)
        self.processID = processID
        self.name = name
        self.meta_data = meta_data
        self.start_idx = start_idx
        self.end_idx = end_idx
    def run(self):
        #print("start process: " + self.name)
        generate_label_and_raw(self.name, self.meta_data, self.start_idx, self.end_idx)
        #print("exit process: "+ self.name)
def generate_label_and_raw(processName, meta_data, idx, end_idx):
    while idx < end_idx:
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end_idx)
            output_message = f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}   Preprocessing, progress: {percent}'            
            print(output_message)
        try:
            row = meta_data.iloc[idx]
            particle_idx = int(row['rlnImageName'].split('@')[0])
            mrc_name = row['rlnImageName'].split('@')[1].split('/')[-1][:-1]
            single_slice_name = f'{particle_idx}@@{mrc_name}'
            if not os.path.exists(f'{output_raw_dir}{single_slice_name}.npy'):
                with mrcfile.open(f'{raw_data_dir}{mrc_name}s') as mrc:
                    raw = mrc.data[particle_idx-1]
                np.save(f'{output_raw_dir}{single_slice_name}', raw)
            if not os.path.exists(f'{output_label_dir}{single_slice_name}.npy'):
                rot = row['rlnAngleRot']
                tilt = row['rlnAngleTilt']
                psi = row['rlnAnglePsi']
                xoff = row['rlnOriginX']
                yoff = row['rlnOriginY']
                project_command = f'relion_project --i {mask_path} --o {output_mask_dir}{single_slice_name} --rot {rot} --tilt {tilt} --psi {psi} --xoff {xoff} --yoff {yoff} > /dev/null 2>&1'
                os.system(project_command)
                with mrcfile.open(f'{output_mask_dir}{single_slice_name}') as mrc:
                    mask_proj = mrc.data
                mask_proj.setflags(write=1)
                mask_proj[mask_proj <= threshold] = 2
                mask_proj[mask_proj > threshold] = 1
                fil = np.full((11,11), -1)
                fil[5,5] = 11*11-1
                contour = convolve(mask_proj, fil, mode='reflect') > 0
                mask_proj[contour == True] = 3
                mask_proj = mask_proj.astype('uint8')
                np.save(f'{output_label_dir}{single_slice_name}', mask_proj)
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        idx += 1


# %%
def preprocessing():
    meta_data = starfile.read(meta_data_path)
    meta_data= meta_data[:number_of_particles_to_process]
    meta_data = meta_data.sample(frac=1, random_state=0)
    particles_to_be_done = len(meta_data)
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
        processes.append( myProcess(i, f"Process-{i}", meta_data, start_idx, end_idx))

    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    all_datasets = pd.read_json(train_data_path)
    all_datasets = all_datasets[all_datasets['skip'] != 'True']
    process_number = 10
    for i in range(len(all_datasets)):
        row = all_datasets.iloc[i]
        dataset_name = row['dataset_name']
        mask_path = row['mask_path']
        meta_data_path = row['meta_data_path']
        raw_data_dir = row['raw_data_dir']
        threshold = row['threshold']
        number_of_particles_to_process = row['number_of_particles_to_process']

        output_mask_dir = f'data_for_training/{dataset_name}/mask/'
        output_label_dir = f'data_for_training/{dataset_name}/label/'
        output_raw_dir = f'data_for_training/{dataset_name}/raw/'
        os.system(f'mkdir -p {output_mask_dir}')
        os.system(f'mkdir -p {output_label_dir}')
        os.system(f'mkdir -p {output_raw_dir}')
        preprocessing()
        print(f'Finished processing {dataset_name}.')
    



# %%

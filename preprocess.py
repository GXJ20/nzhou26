# %%
# this script read metadata from data_for_training.json, maps, raw particles, and store 
# raw particles, map projections in the format of npy.
import mrcfile
import starfile
import os
import numpy as np
import multiprocessing
from scipy.ndimage import convolve
from datetime import datetime
import pandas as pd

train_data_path = 'data_for_training.json'

# multi processing class
class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, meta_data, start_idx, end_idx):
        multiprocessing.Process.__init__(self)
        self.processID = processID
        self.name = name
        self.meta_data = meta_data
        self.start_idx = start_idx
        self.end_idx = end_idx
    def run(self):
        generate_label_and_raw(self.name, self.meta_data, self.start_idx, self.end_idx)

# multi process calls this function, read raw data and make projection from map
# processName: process name that are being used to keep track of progress
# meta_data: the star file in final reconstruction of relion, but in dataframe format
# idx: start index to identify which particle we are processing in the metadata file
# end_idx: end index
def generate_label_and_raw(processName, meta_data, idx, end_idx):
    # we iterate from start index to end index
    while idx < end_idx:
        # print out progress
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end_idx)
            output_message = f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}   Preprocessing, progress: {percent}'            
            print(output_message)
        try:
            
            # locate the particle by index
            row = meta_data.iloc[idx]
            # get particle number in that specific micrographs
            particle_idx = int(row['rlnImageName'].split('@')[0])
            # get name of that micrographs
            mrc_name = row['rlnImageName'].split('@')[1].split('/')[-1]#[:-1]
            single_slice_name = f'{particle_idx}@@{mrc_name[:-1]}'
            # save raw image to npy
            if not os.path.exists(f'{output_raw_dir}{single_slice_name}.npy'):
                with mrcfile.open(f'{raw_data_dir}{mrc_name}') as mrc:
                    raw = mrc.data[particle_idx-1]
                np.save(f'{output_raw_dir}{single_slice_name}', raw)
            # if mask_path == 'none':
            #     mask_proj = np.full(raw.shape,2)
            #     np.save(f'{output_label_dir}{single_slice_name}', mask_proj)
            
            if not os.path.exists(f'{output_label_dir}{single_slice_name}.npy'):
                # get other metadata in order to make projection
                rot = row['rlnAngleRot']
                tilt = row['rlnAngleTilt']
                psi = row['rlnAnglePsi']
                xoff = row['rlnOriginX']
                yoff = row['rlnOriginY']
                # make projections from map and metadata, files saved in mrc format
                project_command = f'relion_project --i {mask_path} --o {output_mask_dir}{single_slice_name} --rot {rot} --tilt {tilt} --psi {psi} --xoff {xoff} --yoff {yoff} > /dev/null 2>&1'
                os.system(project_command)
                # save the mrc format into npy format
                with mrcfile.open(f'{output_mask_dir}{single_slice_name}') as mrc:
                    mask_proj = mrc.data
                
                mask_proj.setflags(write=1)
                # set every pixel smaller than the threshold to 2
                mask_proj[mask_proj <= threshold] = 2
                # set every pixel larger than the threshold to 1
                mask_proj[mask_proj > threshold] = 1
                # create a filter for edge detection
                fil = np.full((11,11), -1)
                fil[5,5] = 11*11-1
                # detect contour
                contour = convolve(mask_proj, fil, mode='reflect') > 0
                # set pixels fall in the contour to 3
                mask_proj[contour == True] = 3
                mask_proj = mask_proj.astype('uint8')
                # save npy
                np.save(f'{output_label_dir}{single_slice_name}', mask_proj)
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        idx += 1


# %%
def preprocessing():
    # read metadata from starfile
    meta_data = starfile.read(meta_data_path)
    meta_data= meta_data[:number_of_particles_to_process]
    # randomize metadata
    meta_data = meta_data.sample(frac=1, random_state=0)
    particles_to_be_done = len(meta_data)
    particles_for_each_process = particles_to_be_done // process_number

    processes = []
    for i in range(process_number):
        # split particle set into batches, each batche use one process
        start_idx = i*particles_for_each_process
        end_idx = start_idx+particles_for_each_process
        if i == process_number -1 :
            end_idx = particles_to_be_done
        # print(i)
        # print(start_idx)
        # print(end_idx)
        processes.append( myProcess(i, f"Process-{i}", meta_data, start_idx, end_idx))

    # run all processes in parallel
    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    # read the json file
    all_datasets = pd.read_json(train_data_path)
    all_datasets = all_datasets[all_datasets['skip'] != 'True']
    process_number = 16
    # iterate through all items in the json
    for i in range(len(all_datasets)):
        row = all_datasets.iloc[i]
        dataset_name = row['dataset_name']
        mask_path = row['mask_path']
        meta_data_path = row['meta_data_path']
        raw_data_dir = row['raw_data_dir']
        threshold = row['threshold']
        number_of_particles_to_process = row['number_of_particles_to_process']

        # set output directories for every category
        output_mask_dir = f'/storage_data/zhou_Ningkun/workspace/data_particleSeg/data_for_training/{dataset_name}/mask/'
        output_label_dir = f'/storage_data/zhou_Ningkun/workspace/data_particleSeg/data_for_training/{dataset_name}/label/'
        output_raw_dir = f'/storage_data/zhou_Ningkun/workspace/data_particleSeg/data_for_training/{dataset_name}/raw/'
        os.system(f'mkdir -p {output_mask_dir}')
        os.system(f'mkdir -p {output_label_dir}')
        os.system(f'mkdir -p {output_raw_dir}')
        preprocessing()
        print(f'Finished processing {dataset_name}.')

# %%

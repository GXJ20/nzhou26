# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import mrcfile
import starfile
import os
import numpy as np
import multiprocessing
from scipy.ndimage import convolve

#your input here
project_dir = '20s_10025'
mask_path = f"{project_dir}/MaskCreate/job029/mask.mrc"
meta_data_path = f'{project_dir}/Refine3D/job026/run_data.star'
threshold = 10
process_number = 30


output_data_dir = f'train_particleSeg/{project_dir}/mask_mrcs/'
os.system(f'mkdir -p {output_data_dir}')

class myProcess (multiprocessing.Process):
    def __init__(self, processID, name, grouped, start_idx, end_idx):
        multiprocessing.Process.__init__(self)
        self.processID = processID
        self.name = name
        self.grouped = grouped
        self.start_idx = start_idx
        self.end_idx = end_idx
    def run(self):
        print("start process: " + self.name)
        generate_mask_stack(self.name, self.grouped, self.start_idx, self.end_idx)
        print("exit process: "+ self.name)
def generate_mask_stack(processName, grouped, idx, end_idx):
    while idx < end_idx:
        try:
            name = grouped[idx][0]
            mrc_slices = []
            for index, row in grouped[idx][1].iterrows():
                particle_idx = row['rlnImageName'].split('@')[0]
                mrc_name = row['rlnImageName'].split('@')[1].split('/')[-1][:-1]
                single_slice_name = f'{particle_idx}--{mrc_name}'
                rot = row['rlnAngleRot']
                tilt = row['rlnAngleTilt']
                psi = row['rlnAnglePsi']
                xoff = row['rlnOriginX']
                yoff = row['rlnOriginY']
                os.system(f'relion_project --i {mask_path} --o {output_data_dir}{single_slice_name} --rot {rot} --tilt {tilt} --psi {psi} --xoff {xoff} --yoff {yoff} > /dev/null 2>&1')
                with mrcfile.open(f'{output_data_dir}{single_slice_name}') as mrc:
                    mask_proj = mrc.data
                mask_proj.setflags(write=1)
                mask_proj[mask_proj <= threshold] = 2
                mask_proj[mask_proj > threshold] = 1
                fil = np.full((11,11), -1)
                fil[5,5] = 11*11-1
                contour = convolve(mask_proj, fil, mode='reflect') > 0
                mask_proj[contour == True] = 3
                mask_proj = mask_proj.astype('uint8')
                mrc_slices.append(mask_proj)
                os.system(f'rm {output_data_dir}{single_slice_name}')
            with mrcfile.open(f"{output_data_dir}{name.split('/')[-1]}s", mode='w+') as mrc_stack:
                mrc_stack.set_data(np.array(mrc_slices))
        except KeyboardInterrupt:
            print('keyboard catched')
            break
        if processName == "Process-0":
            percent = "{:.2%}".format(idx/end_idx)
            print(f"Converting mrcs to png, progress: {percent}")
        idx += 1


# %%
def preprocessing():
    meta_data = starfile.read(meta_data_path)
    grouped = list(meta_data.groupby('rlnMicrographName'))
    mrcs_to_be_done = len(grouped)
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
        processes.append( myProcess(i, f"Process-{i}", grouped, start_idx, end_idx))

    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    preprocessing()
    #training()
    



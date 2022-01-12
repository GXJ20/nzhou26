import starfile
import sys
import pathlib
import os
import random
import numpy as np
star_dir = sys.argv[1]

num_of_particles = os.popen(f'grep mrc {star_dir}/*.star | wc').read().split(' ')[2]
num_of_mrcs = int(os.popen(f'ls {star_dir}/*.star | wc').read().split('   ')[1])
# ratio to add
num_to_add = int(int(num_of_particles) * 1.5)
num_to_add_each = num_to_add // num_of_mrcs
print(f'There are {num_of_particles} particles in {num_of_mrcs} micrographs')
print(f'There will be {num_to_add_each} particles added each micographs')
os.system('mkdir -p messy')
for item in pathlib.Path(star_dir).glob('*.star'):
    meta = starfile.read(item)
    X = meta['rlnCoordinateX'].to_numpy(dtype='float32')
    Y = meta['rlnCoordinateY'].to_numpy(dtype='float32')
    first_row = meta.iloc[0].copy()
    i = 0
    while i < num_to_add_each:
        margins = 100
        coordx = random.randint(0+margins,5760-margins)
        coordy = random.randint(0+margins,4092-margins)
        distance = np.power(np.subtract(coordx, X), 2) + np.power(np.subtract(Y,coordy),2)
        distance = np.sqrt(distance)
        if distance.min() > 120:
            
            first_row.at['rlnCoordinateX']=coordx
            first_row.at['rlnCoordinateY']=coordy
            first_row.at['rlnGroupName'] = 'group_111'
            i += 1
            meta = meta.append(first_row)
            X = np.append(X,coordx)
            Y = np.append(Y,coordy)
            #print(f'adding particle{i} in {item.name}')
    starfile.write(meta, f'messy/{item.name}')
    print(f'Wrote to {item.name}')

import h5py
import re
import os
import sys

sys.path.append(os.path.abspath('../../'))
from common.arydmp import arydmp as ad

def read_hdf5(rpath, wpath):
    if not os.path.isdir(f'./{weights_dirname}'):
        os.mkdir(f'./{weights_dirname}')

    with h5py.File(rpath, 'r') as f: # open file
        keys = []
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                write_file = f[key].name
                write_file = re.sub(r'^\/', '', write_file)
                write_file = re.sub(r'\/', '_', write_file)
                write_file = re.sub(':0$', '', write_file)
                array_name = write_file
                write_file = re.sub('^', f'./{wpath}/', write_file)
                write_file = re.sub('$', '.h', write_file)
                try:
                    if re.search('optimizer_weights', write_file): raise Exception
                    ad.array_dump(f[(key)], write_file, array_name)
                except Exception:
                    pass

if __name__ == '__main__':
    model_filename = 'cats_dogs_giraffes_elephants_lions_classification.h5'
    weights_dirname = 'weights'
    read_hdf5(model_filename, weights_dirname)

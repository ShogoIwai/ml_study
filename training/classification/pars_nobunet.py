import h5py
import numpy as np
import re
import os

model_filename = 'cats_dogs_giraffes_elephants_lions_classification.h5'
weights_dirname = 'weights'

def read_hdf5(rpath, wpath):
    if not os.path.isdir(f'./{weights_dirname}'):
        os.mkdir(f'./{weights_dirname}')

    with h5py.File(rpath, 'r') as f: # open file
        keys = []
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                wpath = f[key].name
                wpath = re.sub(r'^\/', '', wpath)
                wpath = re.sub(r'\/', '_', wpath)
                wpath = re.sub(':0$', '', wpath)
                arry_name = wpath
                wpath = re.sub('^', f'./{weights_dirname}/', wpath)
                wpath = re.sub('$', '.h', wpath)
                try:
                    if re.search('optimizer_weights', wpath): raise Exception
                    np.arry = f[(key)]
                    disp = f'generating {wpath} ..., shape={np.arry.shape}, dtype={np.arry.dtype}'
                    print (disp)
                    header = '// '
                    header += disp
                    header += '\n'
                    header += f'float {arry_name}[] = '
                    header += '{'
                    np.savetxt(wpath, np.arry, header=header, footer='};', delimiter=',', newline=',\n', comments='')
                except Exception:
                    pass

if __name__ == '__main__':
    read_hdf5(model_filename, weights_dirname)

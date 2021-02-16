import h5py
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
                wfile = f[key].name
                wfile = re.sub(r'^\/', '', wfile)
                wfile = re.sub(r'\/', '_', wfile)
                wfile = re.sub(':0$', '.txt', wfile)
                try:
                    if re.search('^optimizer_weights', wfile): raise Exception
                    print (f'generating {wfile} ...')
                    ofs = open(f'./{wpath}/{wfile}', mode='w')
                    print(f'{f[key].name}, {f[(key)]}')
                    # ofs.write(f[(key)])
                    ofs.close
                except Exception:
                    pass

if __name__ == '__main__':
    read_hdf5(model_filename, weights_dirname)

from argparse import ArgumentParser
import h5py
import re
import os
import sys

sys.path.append(os.path.abspath('../../'))
from common.arydmp import arydmp as ad

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--int', help=':int input flag', action='store_true') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.int: opts.update({'int':args.int})

def read_hdf5(rfile, wpath, fflag=True):
    if not os.path.isdir(f'{wpath}'):
        os.mkdir(f'{wpath}')

    with h5py.File(f'{rfile}', 'r') as f: # open file
        keys = []
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                write_file = f[key].name
                write_file = re.sub(r'^\/', '', write_file)
                write_file = re.sub(r'\/', '_', write_file)
                write_file = re.sub(':0$', '', write_file)
                array_name = write_file
                write_file = re.sub('^', f'{wpath}/', write_file)
                write_file = re.sub('$', '.h', write_file)
                try:
                    if re.search('optimizer_weights', write_file): raise Exception
                    ad.array_dump(f[(key)], write_file, array_name, fflag)
                except Exception:
                    pass

if __name__ == '__main__':
    parseOptions()
    model_filename = './cats_dogs_giraffes_elephants_lions_classification.h5'
    weights_dirname = './weights'
    if 'int' in opts.keys():
        read_hdf5(model_filename, weights_dirname, fflag=False)
    else:
        read_hdf5(model_filename, weights_dirname, fflag=True)

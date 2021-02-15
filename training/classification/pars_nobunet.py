import h5py

filename = 'cats_dogs_giraffes_elephants_lions_classification.h5'
def read_hdf5(path):
    keys = []
    weights = {}
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                print(f[key].name)
                weights[f[key].name] = f[key].value
    return weights

if __name__ == '__main__':
    weights = read_hdf5(filename)

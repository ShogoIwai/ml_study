import numpy as np

def array_dump(array, fname, aname):
    array = np.array(array)
    disp = f'generating {fname} ..., shape={array.shape}, dtype={array.dtype}'
    print (disp)
    array = array.flatten()
    header = '// ' + disp + '\n' + f'float {aname}[] = ' + '{' + '\n'
    ofs = open(fname, mode='w')
    ofs.write(header)
    for i in range(array.size):
        ofs.write(f'{array[i]}')
        if (i != (array.size-1)): ofs.write(',\n')
    ofs.write('};')
    ofs.close()

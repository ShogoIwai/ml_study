import numpy as np

def array_dump(array, fname, aname):
    array = np.array(array)
    array_shape = array.shape
    array_dtype = array.dtype
    array = array.flatten()
    array_size = array.size
    disp = f'generating {fname} ..., shape={array_shape}->{aname}[{array_size}], dtype={array_dtype}'
    print (disp)
    header = '// ' + disp + '\n' + f'const float {aname}[] = ' + '{' + '\n'
    ofs = open(fname, mode='w')
    ofs.write(header)
    for i in range(array.size):
        ofs.write(f'{array[i]}')
        if (i != (array.size-1)): ofs.write(',\n')
    ofs.write('};')
    ofs.close()

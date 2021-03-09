import numpy as np

def array_dump(array, fname, aname, iflag=False):
    array = np.array(array)
    array_shape = array.shape
    array_dtype = array.dtype
    array = array.flatten()
    array_size = array.size
    disp = f'generating {fname} ..., shape={array_shape}->{aname}[{array_size}], dtype={array_dtype}'
    print (disp)
    if (iflag):
        header = '// ' + disp + '\n' + f'const int {aname}[] = ' + '{' + '\n'
    else:
        header = '// ' + disp + '\n' + f'const float {aname}[] = ' + '{' + '\n'
    ofs = open(fname, mode='w')
    ofs.write(header)
    for i in range(array.size):
        if (iflag):
            # convint = int(array[i])
            convint = int(array[i]*256)
            ofs.write(f'{convint}')
        else:
            ofs.write(f'{array[i]}')
        if (i != (array.size-1)): ofs.write(',\n')
    ofs.write('};')
    ofs.close()

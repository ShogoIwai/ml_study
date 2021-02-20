import numpy as np

def array_dump(array, fname, aname):
    array = np.array(array)
    disp = f'generating {fname} ..., shape={array.shape}, dtype={array.dtype}'
    print (disp)
    array = array.flatten()
    header = '// ' + disp + '\n' + f'float {aname}[] = ' + '{'
    np.savetxt(fname, array, header=header, footer='};', delimiter=',', newline=',\n', comments='')

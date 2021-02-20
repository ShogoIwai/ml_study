import numpy as np

def array_dump(array, fname, aname):
    np.array = array
    disp = f'generating {fname} ..., shape={np.array.shape}, dtype={np.array.dtype}'
    print (disp)
    header = '// ' + disp + '\n' + f'float {aname}[] = ' + '{'
    np.savetxt(fname, np.array, header=header, footer='};', delimiter=',', newline=',\n', comments='')

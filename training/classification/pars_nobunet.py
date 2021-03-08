from argparse import ArgumentParser
import h5py
from tflite.Model import Model
import re
import os
import sys

sys.path.append(os.path.abspath('../../'))
from common.arydmp import arydmp as ad

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--mdl', help=':specify model file name') # use action='store_true' as flag
    argparser.add_argument('--int', help=':int input flag', action='store_true') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.mdl: opts.update({'mdl':args.mdl})
    if args.int: opts.update({'int':args.int})

def read_tflite_model(file):
    buf = open(file, "rb").read()
    buf = bytearray(buf)
    model = Model.GetRootAsModel(buf, 0)
    return model

def print_model_info(model):
    version = model.Version()
    print("Model version:", version)
    description = model.Description().decode('utf-8')
    print("Description:", description)
    subgraph_len = model.SubgraphsLength()
    print("Subgraph length:", subgraph_len)

def print_nodes_info(model):
    # what does this 0 mean? should it always be zero?
    subgraph = model.Subgraphs(0)
    operators_len = subgraph.OperatorsLength()
    print('Operators length:', operators_len)

    from collections import deque
    nodes = deque(subgraph.InputsAsNumpy())

    STEP_N = 0
    MAX_STEPS = operators_len
    print("Nodes info:")
    while len(nodes) != 0 and STEP_N <= MAX_STEPS:
        print("MAX_STEPS={} STEP_N={}".format(MAX_STEPS, STEP_N))
        print("-" * 60)

        node_id = nodes.pop()
        print("Node id:", node_id)

        tensor = subgraph.Tensors(node_id)
        print("Node name:", tensor.Name().decode('utf-8'))
        print("Node shape:", tensor.ShapeAsNumpy())

        # which type is it? what does it mean?
        # type_of_tensor = tensor.Type()
        # print("Tensor type:", type_of_tensor)

        # quantization = tensor.Quantization()
        # min = quantization.MinAsNumpy()
        # max = quantization.MaxAsNumpy()
        # scale = quantization.ScaleAsNumpy()
        # zero_point = quantization.ZeroPointAsNumpy()
        # print("Quantization: ({}, {}), s={}, z={}".format(min, max, scale, zero_point))

        # I do not understand it again. what is j, that I set to 0 here?
        operator = subgraph.Operators(0)
        for i in operator.OutputsAsNumpy():
            nodes.appendleft(i)

        STEP_N += 1

    print("-"*60)

def read_tflite(rfile, wpath, iflag=False):
    if not os.path.isdir(f'{wpath}'):
        os.mkdir(f'{wpath}')

    model = read_tflite_model(rfile)
    print_model_info(model)
    print_nodes_info(model)

def read_hdf5(rfile, wpath, iflag=False):
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
                    ad.array_dump(f[(key)], write_file, array_name, iflag=False)
                except Exception:
                    pass

if __name__ == '__main__':
    parseOptions()
    model_filename = opts['mdl']
    weights_dirname = './weights'
    if 'int' in opts.keys():
        read_tflite(model_filename, weights_dirname, iflag=True)
    else:
        read_hdf5(model_filename, weights_dirname, iflag=False)

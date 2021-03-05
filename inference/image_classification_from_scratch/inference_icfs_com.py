import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import re
import os
import sys

sys.path.append(os.path.abspath('../../'))
from common.arydmp import arydmp as ad

def export_input_image(img_array, iamge_dir, image_file, int_flag):
    if not os.path.isdir(f'{iamge_dir}'):
        os.mkdir(f'{iamge_dir}')
    write_file = f'{iamge_dir}/{image_file}'
    write_file = re.sub('\.jpg$', '.h', write_file)
    ad.array_dump(img_array, write_file, 'input_image', int_flag)

def export_input_layer(model, img_array, layer_dir, sub_dir, layer_names, int_flag):
    if not os.path.isdir(f'{layer_dir}'):
        os.mkdir(f'{layer_dir}')
    if not os.path.isdir(f'{layer_dir}/{sub_dir}'):
        os.mkdir(f'{layer_dir}/{sub_dir}')
    for idx in range(len(layer_names)):
        layer_name = layer_names[idx]
        layer_model = keras.Model(inputs=model.input,
                                  outputs=model.get_layer(layer_name).output)
        predictions = layer_model.predict(img_array)
        write_file = f'{layer_dir}/{sub_dir}/{layer_name}'
        write_file = re.sub('$', '.h', write_file)
        ad.array_dump(predictions, write_file, layer_name, int_flag)

def classification_single(mdlfile, imgdir, image_size, int_flag=False, div=False):
    model = load_model(mdlfile)
    model.summary()

    image_file = str(imgdir)
    img = keras.preprocessing.image.load_img(image_file, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    if (div):
        img_array = img_array / 255.0

    iamge_dir = f'./image'
    export_input_image(img_array, iamge_dir, image_file, int_flag)

    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    layer_dir = f'./layers'
    sub_dir = re.sub('\.jpg$', '', image_file)
    layer_names = ['conv2d',
                   'batch_normalization',
                   'activation',
                   'max_pooling2d',
                   'conv2d_1',
                   'batch_normalization_1',
                   'activation_1',
                   'max_pooling2d_1',
                   'conv2d_2',
                   'batch_normalization_2',
                   'activation_2',
                   'max_pooling2d_2',
                   'conv2d_3',
                   'batch_normalization_3',
                   'activation_3',
                   'max_pooling2d_3',
                   'flatten',
                   'dense',
                   'batch_normalization_4',
                   'activation_4',
                   'dense_1']
    export_input_layer(model, img_array, layer_dir, sub_dir, layer_names, int_flag)

    predictions = model.predict(img_array)
    print(f"[DEBUG] {image_file}:{predictions}")

    # image = cv2.imread(image_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.title(str(float("{:.2f}".format(predict_max))) + f"[{predict_idx}]")

    # plt.axis("off")
    # predict_max = np.amax(predictions[0])
    # predict_idx = np.argmax(predictions[0])
    # plt.imshow(image)
    # plt.show()

def filter(imgdir, ext):
    image_files = glob.glob(f"{imgdir}*/**", recursive=True)

    for image_file in image_files:
        m = re.search(ext, image_file)
        if (m):
            try:
                fobj = open(image_file, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                print(f"Error: {image_file} can not opend!")
                # Delete corrupted image
                os.remove(image_file)

def classification_multi(mdlfile, imgdir, ext, image_size, div=False, figsize=(15, 9)):
    model = load_model(mdlfile)
    model.summary()

    imgdir = re.sub(r'\/$', '', imgdir)
    image_files = glob.glob(f"{imgdir}*/**", recursive=True)

    size = int(len(image_files))
    width = 5
    if (size < width): width = size
    height = (size / width)
    if (size % width): height = height + 1
    f = plt.figure(figsize=figsize)
    i = 0

    for image_file in image_files:
        m = re.search(ext, image_file)
        if (m):
            img = keras.preprocessing.image.load_img(image_file, target_size=image_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            if (div):
                img_array = img_array / 255.0
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)
            print(f"[DEBUG] {image_file}:{predictions}")

            f.add_subplot(width, height, i + 1)
            i = i + 1

            predict_max = np.amax(predictions[0])
            predict_idx = np.argmax(predictions[0])
            plt.title(str(float("{:.2f}".format(predict_max))) + f"[{predict_idx}]")

            plt.axis("off")
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
    
    plt.show()

def classification(mdlfile, imgdir, ext, image_size, int_flag=False, div=False, figsize=(15, 9)):
    if (re.search(ext, imgdir)):
        print('classification_single processing ...')
        classification_single(mdlfile, imgdir, image_size, int_flag, div)
    else:
        print('classification_multi processing ...')
        filter(imgdir, ext)
        classification_multi(mdlfile, imgdir, ext, image_size, int_flag, div)

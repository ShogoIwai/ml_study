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

def classification_single(mdlfile, imgdir, image_size, div=False):
    model = load_model(mdlfile)
    model.summary()

    image_file = str(imgdir)
    img = keras.preprocessing.image.load_img(image_file, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    if (div):
        img_array = img_array / 255.0

    wpath = f'./image'
    if not os.path.isdir(f'{wpath}'):
        os.mkdir(f'{wpath}')
    write_file = f'{wpath}/{image_file}'
    write_file = re.sub('\.jpg$', '.h', write_file)
    ad.array_dump(img_array, write_file, 'input_image')

    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    print(f"[DEBUG] {image_file}:{predictions}")

    predict_max = np.amax(predictions[0])
    predict_idx = np.argmax(predictions[0])
    plt.title(str(float("{:.2f}".format(predict_max))) + f"[{predict_idx}]")
    plt.axis("off")

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

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

def classification(mdlfile, imgdir, ext, image_size, div=False, figsize=(15, 9)):
    if (re.search(ext, imgdir)):
        print('classification_single processing ...')
        classification_single(mdlfile, imgdir, image_size, div)
    else:
        print('classification_multi processing ...')
        filter(imgdir, ext)
        classification_multi(mdlfile, imgdir, ext, image_size, div)

from PIL import Image 
import os, glob
import numpy as np
from PIL import ImageFile
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["dog", "cat", "giraffe", "elephant", "lion"]
num_classes = len(classes)#リストや文字列など様々な型のオブジェクトのサイズ（要素数や文字数）を取得
image_size = 150
num_testdata = 300

X_train = []
# X_test  = []
# y_train = []
# y_test  = []

#enumerate関数：要素のインデックスと要素を同時に取り出す事が出来る。
#for 変数1, 変数2 in enumerate(リスト):
#print(‘{0}:{1}’.format(変数1, 変数2))
#1行目のfor 変数1, 変数2 in enumerate(list):では、listをenumerateで取得できる間
#ずっと、変数1と変数2に代入し続けるfor文を使用。

for index, classlabel in enumerate(classes):
    photos_dir = "./images/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg") #引数に指定されたパターンにマッチするファイルパス名を取得
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        if i < num_testdata:
            # X_test.append(data)
            # y_test.append(index)
            pass
        else:

            # angleに代入される値
            # -20
            # 0
            # 画像を20度ずつ回転
            for angle in range(-20, 20, 20):

                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                # y_train.append(index)
                # FLIP_LEFT_RIGHT　は 左右反転
                #img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                #data = np.asarray(img_trains)
                #X_train.append(data)
                #y_train.append(index)

print(f'X_train={len(X_train)}')
# print(f'X_test={len(X_test)}')
# print(f'y_train={len(y_train)}')
# print(f'y_test={len(y_test)}')
X_train = np.array(X_train)
# X_test  = np.array(X_test)
# y_train = np.array(y_train)
# y_test  = np.array(y_test)

X_train = X_train / 255
# X_test  = X_test / 255
X_train = X_train.astype("float32")
# X_test  = X_test.astype("float32")

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras.models import load_model
import re

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]

if __name__ == '__main__':
    org_model_name = './cats_dogs_giraffes_elephants_lions_classification.h5'
    conv_model_name = re.sub('\.h5$', '.tflite', org_model_name)

    model = load_model(org_model_name)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    tflite_model = converter.convert()
    open(conv_model_name, 'wb').write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=conv_model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    tf_results = model(tf.constant(input_data))

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    for tf_result, tflite_result in zip(tf_results, tflite_results):
        np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)

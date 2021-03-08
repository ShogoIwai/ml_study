import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import re

if __name__ == '__main__':
    org_model_name = './cats_dogs_giraffes_elephants_lions_classification.h5'
    conv_model_name = re.sub('\.h5$', '.tflite', org_model_name)

    model = load_model(org_model_name)
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
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


import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def preprocess(img_path, input_shape):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    img = tf.image.resize(img, input_shape[:2])

    img = preprocess_input(img)
    return img

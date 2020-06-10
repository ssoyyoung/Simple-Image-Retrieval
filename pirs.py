import struct
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from mysql import connectDB
from config import Setting

def preprocess(img_path, input_shape):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    img = tf.image.resize(img, input_shape[:2])
    img = preprocess_input(img)
    return img


def main():
    batch_size = 100
    input_shape = (224, 224, 3)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False

    model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

    # check data type
    if Setting.DATATYPE == "deepfashion":
        fnames = glob.glob('/data/deepfashion*/**/*.jpg', recursive=True)

    elif Setting.DATATYPE == "pirs":
        fnames = connectDB(65000)

    # make a dataset from a numpy array
    list_ds = tf.data.Dataset.from_tensor_slices(fnames)

    st = time.time()
    ds = list_ds.map(lambda x: preprocess(x, input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # time check : num_parallel_calls = -1 >> 0.135s
    # time check : num_parallel_calls = tf.data.experimental.AUTOTUNE >> 0.128s

    dataset = ds.batch(batch_size).prefetch(-1)

    with open('result/'+ Setting.DATATYPE +'/fvecs.bin', 'wb') as f:
        for batch in dataset:
            fvecs = model.predict(batch)

            fmt = f'{np.prod(fvecs.shape)}f'
            f.write(struct.pack(fmt, *(fvecs.flatten())))

    with open('result/'+ Setting.DATATYPE +'/fnames.txt', 'w') as f:
        f.write('\n'.join(fnames)) 



if __name__ == '__main__':
    main()
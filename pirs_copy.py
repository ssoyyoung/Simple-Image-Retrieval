import struct
import glob
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from mysql import connectDB
from config import Setting
from yolo import getBoundingBox, getBoundingBoxNumpy

def preprocess(img_path, input_shape):
    #img = tf.io.read_file(img_path)
    img = cv2.imread('/mnt/piclick/piclick.tmp/AIMG/PIRS-TEST/PRODUCT-SET/2020/03/05/12/1/366929.jpg')
    img = getBoundingBoxNumpy(img,3)
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.image.decode_jpeg(img, channels=input_shape[2])
    img = getBoundingBoxNumpy(img,3)
    
    """ 
    if rb == "ela":
        # Adding Bounding box crop : get bounding box from Elasticsearch
        rawBox = getRawbox(366929)
        rawBox[0],rawBox[1],rawBox[2],rawBox[3] = rawBox[0]*w,rawBox[1]*h,rawBox[2]*w,rawBox[3]*h
        rawBox = [int(i) for i in rawBox]
        img = tf.image.crop_to_bounding_box(
                                        img,
                                        rawBox[1],
                                        rawBox[0],
                                        abs(rawBox[3]-rawBox[1]),
                                        abs(rawBox[2]-rawBox[0]))
    elif rb == "yolov3":
        rawBox = getBoundingBox(img_path)
        img = tf.image.crop_to_bounding_box(
                                        img,
                                        rawBox[1],
                                        rawBox[0],
                                        abs(rawBox[3]-rawBox[1]),
                                        abs(rawBox[2]-rawBox[0]))
    else:
        pass
    """
    img = tf.image.resize(img, input_shape[:2])
    img = preprocess_input(img)
    return img


def vector(datatype):
    batch_size = 100
    input_shape = (224, 224, 3)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False

    model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

    # check data type
    if datatype == "deepfashion":
        fnames = glob.glob('/data/deepfashion*/**/*.jpg', recursive=True)
        print("total"+datatype+"img count :", len(fnames))

    elif datatype == "pirs":
        fnames = connectDB(10)
        print("total"+datatype+"img count :", len(fnames))

    # make a dataset from a numpy array
    list_ds = tf.data.Dataset.from_tensor_slices(fnames)

    st = time.time()
    ds = list_ds.map(lambda x: preprocess(x, input_shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # time check : num_parallel_calls = -1 >> 0.135s
    # time check : num_parallel_calls = tf.data.experimental.AUTOTUNE >> 0.128s
    print("time check", time.time()-st)

    dataset = ds.batch(batch_size).prefetch(-1)

    with open('result/'+ datatype +'/test_fvecs.bin', 'wb') as f:
        for batch in dataset:
            fvecs = model.predict(batch)

            fmt = f'{np.prod(fvecs.shape)}f'
            f.write(struct.pack(fmt, *(fvecs.flatten())))

    with open('result/'+ datatype +'/test_fnames.txt', 'w') as f:
        f.write('\n'.join(fnames)) 

    return "okay"



if __name__ == '__main__':
    vector("pirs")
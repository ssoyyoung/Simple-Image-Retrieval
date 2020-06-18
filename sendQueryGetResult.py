import os
import struct
import glob
import time
import math
import faiss
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from config import Setting
from elastic import getRawbox


# base model loading
input_shape = (224, 224, 3)
base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                            include_top=False,
                                            weights='imagenet')
base.trainable = False

model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

def preprocess(img_path, input_shape):
    ori_img = tf.io.read_file(img_path)
    ori_img = tf.image.decode_jpeg(ori_img, channels=input_shape[2]) # img.shape = (h,w,c)
    print(ori_img.shape)

    # Adding Bounding box crop : get bounding box from Elasticsearch
    h, w = ori_img.shape[0], ori_img.shape[1]
    rawBox = getRawbox(366929)
    rawBox[0],rawBox[1],rawBox[2],rawBox[3] = rawBox[0]*w,rawBox[1]*h,rawBox[2]*w,rawBox[3]*h
    rawBox = [int(i) for i in rawBox]
    img = tf.image.crop_to_bounding_box(
                                    ori_img,
                                    rawBox[1],
                                    rawBox[0],
                                    abs(rawBox[3]-rawBox[1]),
                                    abs(rawBox[2]-rawBox[0]))
    print("ori_img.shape :",ori_img.shape, "img.shape :",img.shape)
    
    img = tf.image.resize(img, input_shape[:2])
    img = preprocess_input(img)
    return img

# get vector with single image
def getVec(img_path):
    img = tf.reshape(preprocess(img_path, input_shape), [1, 224, 224, 3])
    vec = model.predict(img)
    print(vec.shape)
    return vec


def get_index(index_type, dim):
    if index_type == 'hnsw':
        m = 48
        index = faiss.IndexHNSWFlat(dim, m)
        index.hnsw.efConstruction = 128
        return index
    elif index_type == 'l2':
        return faiss.IndexFlatL2(dim)
        
    raise


def populate(index, fvecs, batch_size=1000):
    nloop = math.ceil(fvecs.shape[0] / batch_size)
    for n in range(nloop):
        s = time.time()
        index.add(normalize(fvecs[n * batch_size : min((n + 1) * batch_size, fvecs.shape[0])]))
        print(n * batch_size, time.time() - s)

    return index

def searchVec(imgPath):
    vec = getVec(imgPath)

    dim = 1280
    base_dir = 'result/'+Setting.DATATYPE+'/'
    fvec_file = base_dir +'fvecs.bin'
    index_type = 'hnsw'
    index_file = f'{fvec_file}.{index_type}.index'

    fvecs = np.memmap(fvec_file, dtype='float32', mode='r').view('float32').reshape(-1, dim)

    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        if index_type == 'hnsw':
            index.hnsw.efSearch = 256
    else:
        index = get_index(index_type, dim)
        index = populate(index, fvecs)
        faiss.write_index(index, index_file)

    k = 10

    dists, idxs = index.search(normalize(vec),k)

    imgName = "test4_"

    np.save(base_dir+imgName+"distance.npy", dists)
    np.save(base_dir+imgName+"resultIdx.npy", idxs)

    return "done"
    

if __name__ == '__main__':
    # send query image for search
    searchVec('test4.jpg')



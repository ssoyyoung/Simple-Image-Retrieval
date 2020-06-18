import os
import time
import math
import random
import numpy as np
import json
from sklearn.preprocessing import normalize
import faiss
from config import Setting


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


def createQueryDB(datatype):
    dim = 512
    base_dir = 'result/torch/'
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

    random.seed(2020)
    q_idx = [random.randint(0, fvecs.shape[0]) for _ in range(100)]

    k = 10
    s = time.time()
    dists, idxs = index.search(normalize(fvecs[q_idx]), k)

    np.save(base_dir+"queryIdx.npy", q_idx)
    np.save(base_dir+"distance.npy", dists)
    np.save(base_dir+"resultIdx.npy", idxs)

    print((time.time() - s) / len(q_idx))

    s = time.time()
    for i in q_idx:
        dists, idxs = index.search(normalize(fvecs[i:i+1]), k)
    print((time.time() - s) / len(q_idx))

    return "create"
    

if __name__ == '__main__':
    main()

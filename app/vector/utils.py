import os
import json
import math
import struct
import numpy as np

from config.config import ResultConfig, DataConfig


# default settings 
baseDir = ResultConfig.BASE_DIR
fvecsBin = ResultConfig.FVECS_BIN
fnamesTxt = ResultConfig.FNAMES


def cleanUP(dirPath):
    '''
    Coment : delete all file in dirPath
    `
    @parameter : directory Path [Type : str]
    `
    @return : None
    '''
    for folder in os.scandir(dirPath):
        for files in os.scandir(folder.path):
            os.remove(files.path)
    
    print(f'[DIR] Deleting all files/folders in {dirPath}')


def mkdirSavePath(key):
    '''
    Coment : create directory for saving data (if no directory, create new directory)
    `
    @parameter : category name [Type : str]
    `
    @return : new Save Path [Type: str]
    '''
    savePath = baseDir+key+"/"
    if not os.path.exists(savePath): os.makedirs(savePath)  

    return savePath   


def saveVecs(vecs, key, writeM):
    '''
    Coment : create vec.bin file
    `
    @parameter : (1) stack of Tensor [Type : pytorch 4-dim tensor] > torch.Size([n, 512, 1, 1])
                    (2) key : folder_name, 
                    (3) write_method : [w: write new, a: write continue]
    `
    @return : None
    '''
    savePath = mkdirSavePath(key) + fvecsBin
            
    with open(savePath, writeM+'b') as f:
        fmt = f'{np.prod(vecs.shape)}f'
        f.write(struct.pack(fmt, *(vecs.flatten())))


def saveInfos(info, key, writeM):
    '''
    Coment : create data_info.txt file
    `
    @parameter : (1) crop Img label [Type : list of dict] {'img_path','raw_box','conf','class'}, 
                    (2) key : folder_name, 
                    (3) write_method : [w: write new, a: write continue]
    `
    @return : None
    '''

    savePath = mkdirSavePath(key) + fnamesTxt
    with open(savePath, writeM) as f:
        for line in info:
            f.write(json.dumps(line)+'\n')


def saveNumpy(nparrays, fileNames):
    if len(fileNames) == 1:
        np.save(fileNames, nparrays)
    else:
        for fileName, nparray in zip(fileNames, nparrays):
            np.save(fileName, nparray)


def divideData(totalDataCount, Max):
    div = 0
    if totalDataCount > Max: 
        div = totalDataCount/Max
    div = math.floor(div)
    print(f'[DATA] Max count is {Max}, Divid Count is {div}')

    return div

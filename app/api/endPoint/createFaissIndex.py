from fastapi import APIRouter

from app.dataBase import MySQL
from app.vector.extVector import Vector
from app.faissLib.extFaiss import FaissLib
from app.vector import utils as util
from config.config import DataConfig, ResultConfig

import time

# define router
router = APIRouter()

# difine class
VEC = Vector()
faisss = FaissLib() # conda install faiss-gpu cudatoolkit=10.0 -c pytorch [FAISS INSTALL]

@router.put('/{cate}/{num}')
async def createFaiss(cate, num:int):
    #cate, num, writeM = "WC12", 100, "a"

    util.cleanUP(ResultConfig.BASE_DIR)

    flist = MySQL.getImgList(cate, num)
    yoloResult = VEC.getYoloBox(fnameList=flist, dbCate = "Top")
    print(yoloResult['info'].keys())
    for key in yoloResult['info'].keys():
        print(f'[EXTRACT AND SAVE] Starting {key} Category')
        #if not key == "Dress": continue
        count = len(yoloResult['info'][key])
        st_time = time.time()

        # Save INFO
        util.saveInfos(yoloResult['info'][key], key, ResultConfig.WRITE_MODE)

        Max = 1000
        div = util.divideData(len(yoloResult['info'][key]), Max=Max)

        for d in range(div+1):
            tensorStack = VEC.changePILtoTensorStack(yoloResult['vecs'][key][d*Max:(d+1)*Max])
            extVecs = VEC.extractVec(vecs=tensorStack, model="resnet") # Take Time
            util.saveVecs(extVecs, key, ResultConfig.WRITE_MODE)

        # # create Faiss Index
        faisss.writeIndexFile(key)
        
        endTime = time.time()-st_time

        print(f'[{key}] count : {count} & time : {endTime}')
        del st_time, endTime, count, tensorStack, extVecs


    

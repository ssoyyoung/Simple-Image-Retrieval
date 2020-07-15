from fastapi import APIRouter

from config.config import *
from app.dataBase import MySQL

from app.vector.extVector import Vector
from app.elastic import utils as EU
from app.elastic.elasticsearch import *

# define router
router = APIRouter()

# difine class
VEC = Vector()

@router.put('/{cate}')
async def createVector(cate: str):
    dbData = MySQL.getAllData(cate)
    imgList = [*dbData]
    
    if len(imgList) > DataConfig.MAX_COUNT: div = dbData/DataConfig.MAX_COUNT

    for d in div:
        fromNum = div * DataConfig.MAX_COUNT 
        toNum = (div + 1) * DataConfig.MAX_COUNT 
        print(fromNum, toNum)
        #yoloResult = VEC.getYoloBox(imgList[]) # unpacking generalizations / Unpacking with * works with any object that is iterable
        

    # for key in yoloResult['info'].keys():
    #     if not key == "Dress" : continue

    #     tensorStack = VEC.changePILtoTensorStack(yoloResult['vecs'][key])
    #     extVecs = VEC.extractVec(vecs = tensorStack, model="cgd")
    #     saveIndex = EU.createIndexName(key)

    #     insertBulkData(vecs=extVecs, infos=yoloResult['info'][key], dataDict=dbData, index=saveIndex)









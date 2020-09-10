import time
import base64
from PIL import Image

from fastapi import APIRouter

from app.schema import schema
from app.schema import utils
from app.faissLib import utils as Futil
from app.faissLib.extFaiss import FaissLib
from app.vector.extVector import Vector

from sklearn.preprocessing import normalize

VEC = Vector()
fai = FaissLib()

router = APIRouter()

@router.put('/')
async def searchFaiss(inputData:  schema.Input):
    st_time = time.time()
    imgB64 = inputData.b64Image
    print(f'input data type is {type(imgB64)}')
    decodeImg = utils.decode_b64_to_pil(imgB64)

    res = VEC.getBBoxFromPILImage(decodeImg)
    key = "Top"
    tensorImg = VEC.changePILtoTensorStack(res['vecs'][key])
    extVec = VEC.extractVec(vecs=tensorImg, model="resnet")
    extVec = extVec.cpu().numpy().flatten().reshape(1, -1)

    index = fai.searchFaissIndex(key, extVec)
    imgList = Futil.loadFname(key, index)
    imgList = [img.replace("/mnt/piclick/piclick.tmp/AIMG", "https://storage.cloud.google.com/piclick-ai") for img in imgList]
    
    print(time.time()-st_time)

    return imgList



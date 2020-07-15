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
    print(res)
    tensorImg = VEC.changePILtoTensorStack(res['vecs']['Dress'])
    extVec = VEC.extractVec(vecs=tensorImg, model="resnet")
    extVec = extVec.cpu().numpy().flatten().reshape(1, -1)

    index = fai.searchFaissIndex('Dress', extVec)
    imgList = Futil.loadFname('Dress', index)
    imgList = [img.replace("/mnt/piclick/piclick.tmp/AIMG", "https://storage.cloud.google.com/piclick-ai") for img in imgList]
    
    # imgList = [
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/4/424372.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/12/7/375127.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/12/E/380734.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/6/412454.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/6/427734.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/0/412080.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/4/429556.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/A/432106.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/1/432385.jpg',
    #     'https://storage.cloud.google.com/piclick-ai/PIRS-TEST/PRODUCT-SET/2020/03/05/13/B/434843.jpg']

    print(time.time()-st_time)

    return imgList



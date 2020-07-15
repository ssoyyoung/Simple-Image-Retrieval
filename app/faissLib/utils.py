import json

def loadFname(key, indexList):
    with open(f'resultFile/v1/{key}/fnames.txt', "r") as f:
        fname = f.readlines()
    
    imgList = [json.loads(fname[idx])['img_path'] for idx in indexList[0]]

    return imgList
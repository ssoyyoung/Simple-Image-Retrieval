import json
import os

with open("config/DBinfo.json") as f:
    database = json.load(f)

# 설정

DEBUG = True
MAIN_MODEL: str = "cgd"
MAIN_MODEL_NUM: int = 10

class DataConfig():
    CATE: str = "WC13"
    NUM: int = 500
    WRITE_M: str = "a" # a / w
    MAX_COUNT: int = 10000

class YoloConfig():

    BATCH_SIZE: int = 100
    CONF_THRES: float = 0.2 
    NMS_THRES: float = 0.6
    IMG_SIZE: int = 416

    # NOT CHANGE
    CFG: str = '/fastapi/yolov3/cfg/fashion/fashion_c23.cfg'
    WEIGHTS: str = '/mnt/piclick/piclick.ai/weights/best.pt'
    LABEL_PATH: str = '/fastapi/yolov3/data/fashion/fashion_c23.names'

class CgdConfig():
    ARCHITECTURE: str = 'seresnet50'
    OUTPUT_DIM: int = 1536
    COMBINATION: str = 'GS'
    PRETRAINED: bool = True
    CLASSES: int = 3946

    MODEL_NUM: int = 10 #f'{MAIN_MODEL_NUM}'
    WEIGHTDIR: str = "/mnt/piclick/piclick.ai/weights/"



class ExtractorCofing():
    DIM: int = 512
    BATCH: int = 100
    MODEL: str = "resnet"

class FaissConfig():
    K: int = 10

class ResultConfig():
    BASE_DIR: str = "resultFile/v1/"

    # NOT CHANGE
    FVECS_BIN: str = "fvecs.bin"
    FNAMES: str = "fnames.txt"
    INDEX_TYPE: str = "hnsw"

    WRITE_MODE: str = "a"


class History():
    v1: str = "test trial"

class Setting():
    # Base Img Path
    BASE_IMG_PATH: str = "/mnt/piclick/piclick.tmp/AIMG/"

    # DB Auth info
    DATABASE_HOST: str = database['db']['host']
    DATABASE_USER: str = database['db']['user']
    DATABASE_PWD: str = database['db']['password']
    DATABASE_DB: str = database['db']['db']

    ALL_DATA_QUERY: str = "SELECT * FROM `crawling_list` WHERE STATUS=1"
    PATH_DATA_QUERY: str = "SELECT save_path, save_name FROM `crawling_list` WHERE STATUS=1"

class Elastic():
    # Elastic info
    ELA_HOST: str = database['ela']['host']
    ELA_PORT: str = database['ela']['port']
    ELA_ADDR: str = ELA_HOST+":"+ELA_PORT

    SAVE_INDEX: str = f'{MAIN_MODEL}{MAIN_MODEL_NUM}'
    SAVE_FIELD: str = f'{MAIN_MODEL}{MAIN_MODEL_NUM}_vector'
    GS_BUCKET: str = "https://storage.cloud.google.com"


import json
import base64
import numpy as np
from datetime import datetime

from config.config import Elastic

def es_parsing(res, rb_list=[], cate_list=[], multi=True):
    es_total = []
    res = eval(res.replace('false', 'False'))

    if multi: #boxes in one img
        for idx in range(len(res['responses'])):
            es = []
            es_obj = {}
            for count in range(len(res['responses'][idx]['hits']['hits'])):
                data = res['responses'][idx]['hits']['hits'][count]['_source']
                data['box_state'] = data.pop('box_statue')
                data['_score'] = res['responses'][idx]['hits']['hits'][count]['_score']
                es.append(data)
            es_obj['input_raw_box'] = rb_list[idx]
            es_obj['cat_key'] = cate_list[idx]
            es_obj['search_result'] = es
            es_total.append(es_obj)
    else: #one img
        es = []
        es_obj = {}
        for count in range(len(res['hits']['hits'])):
            data = res['hits']['hits'][count]['_source']
            data['_score']=res['hits']['hits'][count]['_score']
            data['raw_box']=None
            es.append(data)
        es_obj['input_raw_box'] = None
        es_obj['cat_key'] = None
        es_obj['search_result'] = es
        es_total.append(es_obj)

    return es_total


def createIndexName(cate):
    return f'{Elastic.SAVE_INDEX}_{cate.lower()}'


dataFormat = "%Y-%m-%d %H:%M:%S"
float32 = np.dtype('>f4')


def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(float32)).decode("utf-8")
    return base64_str


def utc_time():
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=float32).tolist()


def addMapping(save=False):
    
    with open("mapping/baseMapping.json", "r") as f:
        baseM = json.load(f)

    new = Elastic.SAVE_FIELD
    baseM['mappings']['properties'][new] = {
                                            "type" : "binary",
                                            "doc_values" : True
                                            }
    if save:
        with open(f'mapping/{new}_mapping.json', "w") as f:
            json.dump(baseM, f, ensure_ascii=False, indent='\t')
        return
    else:
        return baseM

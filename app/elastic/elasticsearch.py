import time
from elasticsearch import Elasticsearch, helpers

from config.config import Elastic
from app.elastic import utils as EU


es = Elasticsearch(Elastic.ELA_ADDR, timeout=30, max_retries=10, retry_on_timeout=True)

def createIndex(indexName):
    
    mapping = EU.addMapping(save=False)
    
    if not type(indexName) == list: indexName = [indexName]
    for index in indexName:
        es.indices.create(index=index, body=mapping)
        print(f'[Elastic] "{index}" index create!')
    

def checkIndexExsist(index):
    return es.indices.exists(index=index)


def insertBulkData(vecs, infos, dataDict, index):
    # datadict = id, au_id, cat_key, i_key, img_url, click_url, img_path, group_id, status, gs_bucket
    if not checkIndexExsist(index=index): 
        createIndex(indexName=index)

    docs = []
    es_time = time.time()
    for vec, info in zip(vecs, infos):
        dDict = dataDict[info['img_path']]
        docs.append({
            "_index": index,
            "_source": {
                "id": dDict[0],
                "au_id": dDict[1],
                "cat_key": dDict[2],
                "i_key": dDict[3],
                "img_url": dDict[4],
                "click_url": dDict[5],
                "img_path": dDict[6],
                "group_id": dDict[7],
                "status": dDict[8],
                "gs_bucket" : f'{Elastic.GS_BUCKET}{dDict[9][4:]}',
                "raw_box": info['raw_box'],
                "yolo_label": info['class'],
                Elastic.SAVE_FIELD : EU.encode_array(vec),
                "@timestamp": EU.utc_time()
            }
        })
    
    helpers.bulk(es, docs)
    print('[TIME] Interface time for sending to elastic', time.time() - es_time)

from elasticsearch import Elasticsearch
from config import Setting

es = Elasticsearch(Setting.ELA_ADDR, timeout=30, max_retries=10, retry_on_timeout=True)

# INPUT : id // OUTPUT : raw_box
def getRawbox(id):
    
    index = "pirs*"
    body = {
        "_source": "raw_box",
        "query": {
            "match": {
                "id" : str(id)
            }
        }
    }

    res = es.search(index=index, body=body)
    rawBox = res['hits']['hits'][0]['_source']['raw_box']

    return rawBox

if __name__ == "__main__":
    getRawbox(366929)


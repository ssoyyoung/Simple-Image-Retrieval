from elasticsearch import elasticsearch


es = Elasticsearch("http://47.56.200.94:9200")

def getRawbox(idList):
    # TODO : create function for get rawbox in elasticsearch database
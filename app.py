from flask import Flask
import pirs
import pirsSearch
import sendQueryGetResult

app = Flask(__name__)

datatype = "deepfashion"

@app.route('/')
def checkFlask():
    return "Hello Flask"

@app.route('/getVector/<datatype>')
def getVector(datatype):
    return pirs.vector(datatype)

@app.route("/queryResult")
def queryResult():
    return pirsSearch.createQueryDB(datatype)

@app.route("/searchVec/<imgPath>")
def searchVec(imgPath):
    return sendQueryGetResult.searchVec(imgPath)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8002, debug=True)


    # tips
    # debug mode on => debug=True
    # allow access all host => host='0.0.0.0'
    # path parameters => <path parameter>
    # GET은 모든 파라미터를 url로 보내는 것(눈에 보임) : 작은양의 데이터 전송
    # POST는 전달하려는 정보가 HTTP body에 포함되어 전달되는 것(눈에 보이지 않음) : 큰양의 데이터 전송
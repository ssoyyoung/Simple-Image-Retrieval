import os
import cv2 # pip install opencv-python
import json
from PIL import Image # pip install Pillow
import pymysql
from config import Setting


def checkImg(imgList):
    TimgList = []
    for img_path in imgList:
        if not os.path.isfile(img_path): 
            continue
        if cv2.imread(img_path) is None: 
            continue
        try: 
            Image.open(img_path).convert('RGB')
        except: 
            continue
        TimgList.append(img_path)

    return TimgList


def connectDB(data_count):
    base_img_path = "/mnt/piclick/piclick.tmp/AIMG/"

    conn = pymysql.connect( 
                        host=Setting.DATABASE_HOST,
                        user=Setting.DATABASE_USER,
                        password=Setting.DATABASE_PWD,
                        db=Setting.DATABASE_DB
                        )
 
    sql_query = "SELECT save_path, save_name FROM `crawling_list` WHERE STATUS=1 AND cat_key='WC13' LIMIT "+str(data_count)

    curs = conn.cursor()
    curs.execute(sql_query)
    data = curs.fetchall()
    
    # change img_path
    imgList = [base_img_path+path+"/"+name for path, name in data]
    imgList = checkImg(imgList)
    print("total file count is...", len(imgList))

    curs.close()
    conn.close()

    return imgList


if __name__ == "__main__":
    connectDB(50)
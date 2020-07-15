import glob
import time
import struct
import json
from tqdm import tqdm
import numpy as np
from PIL import Image

from config.config import YoloConfig, ResultConfig, DataConfig
from app.model.dataloader import YoloDataset, YoloImg

from app.model.utils import *
from app.vector.utils import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from collections import defaultdict

# Model
from app.model.yolov3 import YoloV3
from app.model.extModel import Resnet18, Cgd


# define Model
yoloM = YoloV3()
resnet18 = Resnet18()
cgd = Cgd()

class Vector():
    def __init__(self):
         # default settings 
        self.conf_thres = YoloConfig.CONF_THRES
        self.nms_thres = YoloConfig.NMS_THRES
        self.batch_size = YoloConfig.BATCH_SIZE
        self.image_size = YoloConfig.IMG_SIZE

        self.cate = DataConfig.CATE
        self.num = DataConfig.NUM

        # tranforms
        self.transform = transforms.Compose([
                transforms.ToTensor()
            ])    

        # Model
        # self.yoloM = YoloV3()
        # self.resnet18 = Resnet18()
        # self.cgd = Cgd()


    def getYoloBox(self, fnameList, dbCate = "Dress"):
        '''
        @parameter : image Path [Type : List]
        `
        @return : crop img tensor vector & info [Type : dict]
               >> return['info'][label] = [{'img_path', 'raw_box', 'conf', 'class'}]
                        ['vecs'][label] = [PIL.Image : RGB mode]   
        '''

        fnames = fnameList
        yoloDataset = YoloDataset(
                            path=fnames,     
                            label=[], 
                            img_size=self.image_size, 
                            batch_size=self.batch_size, 
                            transform=self.transform)
        yoloDataLoader = DataLoader(
                            yoloDataset, 
                            batch_size=self.batch_size,
                            shuffle=False, 
                            num_workers=4,
                            collate_fn=yoloDataset.collate_fn)
        # create defaultdict for return
        res = defaultdict(lambda: defaultdict(list))

        print(f'[YOLO] Detecting Bounding Box')
        # iterate for box data
        for batch, (imgs, img0s, paths, labels, shapes) in enumerate(tqdm(yoloDataLoader)):
            #print(f'######## {batch} ########')
            torch.cuda.empty_cache()

            with torch.no_grad():   
                imgs = imgs.to(yoloM.device)   
                _, _, height, width = imgs.shape        

                pred = yoloM.model(imgs)[0]
                pred = non_max_suppression(pred, conf_thres=self.conf_thres, nms_thres=self.nms_thres)

            for i, det in enumerate(pred):
                img0shape = shapes[i]
                if det is not None and len(det):
                    # make original size
                    det[:, :4] = scale_coords(imgs.size()[2:], det[:, :4], img0shape).round() 

                    for *xyxy, conf, lab in det:
                        saveLabel = yoloM.names[int(lab)]
                        if not saveLabel == dbCate : continue
                        xyxy = [int(x) for x in xyxy]

                        res['info'][saveLabel].append({
                                                    "img_path" : paths[i], 
                                                    "raw_box" : xyxy, 
                                                    "conf" : round(float(conf),3), 
                                                    "class" : saveLabel
                                                    })
                        
                        cropBox = letterbox(img0s[i].crop(xyxy), desired_size=224)
                        res['vecs'][saveLabel].append(cropBox)


            with open("yoloResult.json", "a") as f:
                json.dumps(res, indent='\t')


        return res

    
    def getBBoxFromPILImage(self, pilImg):
        pilImg = Image.open(pilImg).convert('RGB')
        
        img = letterbox(pilImg, desired_size=416)

        img = self.transform(img).unsqueeze(0)
        print(img.size())

        torch.cuda.empty_cache()

        # create defaultdict for return
        res = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():   
            img = img.to(yoloM.device)   
            _, _, height, width = img.shape        

            pred = yoloM.model(img)[0]
            pred = non_max_suppression(pred, conf_thres=self.conf_thres, nms_thres=self.nms_thres)

        for i, det in enumerate(pred):
            img0shape = pilImg.size[1], pilImg.size[0]
            if det is not None and len(det):
                # make original size
                det[:, :4] = scale_coords(img.size()[2:], det[:, :4], img0shape).round() 

                for *xyxy, conf, lab in det:
                    saveLabel = yoloM.names[int(lab)]
                    xyxy = [int(x) for x in xyxy]

                    res['info'][saveLabel].append({
                                                "raw_box" : xyxy, 
                                                "conf" : round(float(conf),3), 
                                                "class" : saveLabel
                                                })
                    
                    cropBox = letterbox(pilImg.crop(xyxy), desired_size=224)
                    res['vecs'][saveLabel].append(cropBox)
            
        return res



    
    def changePILtoTensorStack(self, PILlist):
        '''
        @parameter : PIL image list [Type : List of PIL image]
        `
        @return : stack of Tensor [Type : 4-dim tensor]
        '''
        tensorStack = torch.stack([self.transform(PIL) for PIL in PILlist], 0)
        
        return tensorStack


    def extractVec(self, vecs, model):
        '''
        @parameter : stack of Tensor [Type : pytorch 4-dim tensor] > torch.Size([n, 3, 224, 224])
        `
        @return : stack of Tensor [Type : pytorch 4-dim tensor] > torch.Size([n, 512, 1, 1])
        '''

        print(f'[VECTOR] Extract Vector Using "{model}"')
        stTime = time.time()
        if model == "resnet":
            vecs = resnet18.featureExt(vecs)
        elif model == "cgd":
            vecs = cgd.featureExt(vecs)

        print(f'[VECTOR] Extract Vector Time Check : {time.time()-stTime}')
        
        return vecs         


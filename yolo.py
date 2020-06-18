from yolov3.models import *
from yolov3.pirs_utils_v2 import *
from torch.utils.data import DataLoader

import torch
from torchvision import transforms

import cv2
import numpy as np

img_size = 416

cfg = '/fastapi/yolov3/cfg/fashion/fashion_c23.cfg'
weights = '/mnt/piclick/piclick.ai/weights/best.pt'
names_path = '/fastapi/yolov3/data/fashion/fashion_c23.names'

device = torch_utils.select_device('')
model = Darknet(cfg, arc='default').to(device).eval()
model.load_state_dict(torch.load(weights, map_location=device)['model'])
names = load_classes(names_path)

# Only one Image using imgPath
def getBoundingBox(imgPath): 
    img = cv2.imread(imgPath)
    img, img0 = transfer_b64(img, mode='square')

    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    nms_pred = non_max_suppression(pred, conf_thres=0.2, nms_thres=0.5)

    res = []

    # Process detections
    for i, det in enumerate(nms_pred):
        if det is not None and len(det):
            im0shape = img0.shape
            print(im0shape)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

            for *xyxy, conf, cls in det:
                #if conf < 0.5: continue
                xyxy = [int(x) for x in xyxy]

                # box가 이미지를 벗어나는 것이 있는지 확인 후 CHECK
                res.append([xyxy, round(float(conf),3), names[int(cls)]])
            
            return res

        else:
            return None

# Only one Image using numpy
def getBoundingBoxNumpy(imgArray, anw): 
    img, img0 = transfer_b64(imgArray, mode='square')

    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    nms_pred = non_max_suppression(pred, conf_thres=0.2, nms_thres=0.5)

    res = []

    # Process detections
    for i, det in enumerate(nms_pred):
        if det is not None and len(det):
            im0shape = img0.shape
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

            for *xyxy, conf, cls in det:
                if int(cls) == anw:
                    xyxy = [int(x) for x in xyxy]
                    print("box")

                    # return crop numpy array
                    return img0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    #res.append([xyxy, round(float(conf),3), names[int(cls)]])
                else:
                    continue
    if len(res) == 0:
        print("full")
        return img0


if __name__ == "__main__":
    res = getBoundingBox("test3.jpg")
    print(res)
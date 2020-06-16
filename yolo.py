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

def getBoundingBox(imgPath): 
    img = cv2.imread(imgPath)
    img, img0 = transfer_b64(img, mode='square')

    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    nms_pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

    res = []

    # Process detections
    for i, det in enumerate(nms_pred):
        if det is not None and len(det):
            im0shape = img0.shape
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

            for *xyxy, conf, cls in det:
                #if conf < 0.5: continue
                xyxy = [int(x) for x in xyxy]
                res.append([xyxy, round(float(conf),3), names[int(cls)]])
            
            return res

        else:
            return None


if __name__ == "__main__":
    res = getBoundingBox("test4.jpg")
    print(res)
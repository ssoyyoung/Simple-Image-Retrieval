from yolov3.models import *
from yolov3.pirs_utils_v2 import *
from torch.utils.data import DataLoader
from PIL import Image

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

def letterbox(im, desired_size = 416):
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size), color="gray")
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_im

def getBox(inputImg,anw):
    #image = Image.open(imgPath).convert('RGB')
    image = letterbox(inputImg, 416)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img = transform(image).to(device)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    nms_pred = non_max_suppression(pred, conf_thres=0.2, nms_thres=0.5)

    res = []
    # Process detections
    for i, det in enumerate(nms_pred):
        if det is not None and len(det):
            im0shape = inputImg.size[1],inputImg.size[0]
            det[:, :4] = scale_coords(img.size()[2:], det[:, :4], im0shape).round()  # originial

            for *xyxy, conf, cls in det:
                #if conf < 0.5: continue
                xyxy = [int(x) for x in xyxy] # x1,y1,x3,ye

                if names[int(cls)] == anw:
                    res.append([xyxy, round(float(conf),3), names[int(cls)]])
    
    torch.cuda.empty_cache()
    
    return res


from mysql import connectDB

import glob
import struct
import json
import numpy as np
from PIL import Image

from yolov3.models import *
from yolov3.pirs_utils_v2 import *

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


class YoloData(Dataset):
    def __init__(self, path, label, img_size, batch_size, transform):
        self.img_files = [x for x in path if "." + x.split('.')[-1].lower() in img_formats]
        self.label = label
        self.n = len(self.img_files)
        assert self.n > 0, 'No images found in %s' % (path)
        self.transform = transform
        self.imgs = [None] * self.n
        self.batch = np.floor(np.arange(self.n) / batch_size).astype(np.int)  # batch index of image
        self.img_size = img_size
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): # PIL size = w, h
        imgPath = self.img_files[idx]
        img0 = Image.open(imgPath).convert('RGB')
        w0, h0 = img0.size
        img = letterbox(img0, 416)
        
        tran_img = self.transform(img)
        return tran_img, img, imgPath, self.label, (h0, w0)
    
    @staticmethod
    def collate_fn(batch):
        img, img0, path, label, shapes = list(zip(*batch))  # transposed
        return torch.stack(img, 0), img0, path, label, shapes


def letterbox(im, desired_size = 416):
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size), color="gray")
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_im



transform = transforms.Compose([
        transforms.ToTensor()
    ])    

batch_size = 100

fnames = connectDB(1000)
#fnames = glob.glob("testImg/*.*")
pirs_dataset = YoloData(path=fnames, label=[], img_size=416, batch_size=batch_size, transform=transform)

dataloader = DataLoader(pirs_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=1,
                        collate_fn=pirs_dataset.collate_fn)

cfg = '/fastapi/yolov3/cfg/fashion/fashion_c23.cfg'
weights = '/mnt/piclick/piclick.ai/weights/best.pt'
names_path = '/fastapi/yolov3/data/fashion/fashion_c23.names'

device = torch_utils.select_device('')
model = Darknet(cfg, arc='default').to(device).eval()
model.load_state_dict(torch.load(weights, map_location=device)['model'])
names = load_classes(names_path)

model_ft = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])


for batch, (imgs, img0s, paths, labels, shapes) in enumerate(dataloader):
    torch.cuda.empty_cache()
    
    print(imgs.shape)
    with torch.no_grad():   
        imgs = imgs.to(device)   
        _, _, height, width = imgs.shape        

        pred = model(imgs)[0]
        pred = non_max_suppression(pred, conf_thres=0.2, nms_thres=0.6)

        del imgs

    # Process detections
    res = []
    info = []
    for i, det in enumerate(pred):
        im0shape = shapes[i]
        if det is not None and len(det):
            #print(det[:, :4])
            #make origin size
            #det[:, :4] = scale_coords(imgs.size()[2:], det[:, :4], im0shape).round()  # originial
            
            xyxyList = []
            for *xyxy, conf, cls in det:
                #if conf < 0.5: continue
                #if not int(cls) == labels[i]:
                #    continue
                xyxy = [int(x) for x in xyxy] # x1,y1,x3,y3
                #print([xyxy, round(float(conf),3), names[int(cls)]])
                info.append([paths[i], xyxy, round(float(conf),3), names[int(cls)]])
                res.append(letterbox(img0s[i].crop(xyxy), desired_size=224))
    
    result = [transform(x) for x in res]
    result = torch.stack(result, 0)
    print(result.shape)
    print("============")
            
    #fvecs = feature_extractor(result)
    #print(fvecs.shape)

    with open('result/torch/yolo.bin', 'ab') as f:
        fvecs = feature_extractor(result)

        fmt = f'{np.prod(fvecs.shape)}f'
        f.write(struct.pack(fmt, *(fvecs.flatten())))

    with open('result/torch/yolo.txt', 'a') as f:
        for line in info:
            #f.write(str(line)+'\n') 
            f.write(json.dumps(line)+'\n')
    
    del result


import torchvision.transforms as transforms
import torchvision.models as models
from skimage import io
import torch
import glob
import time
from mysql import connectDB
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct

class Pirs(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])
        print(idx,"origin shape",image.size)
        tran_img = self.transform(image)
        print(idx,"trans shape", tran_img.shape)

        return tran_img


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


#fnames = connectDB(65000)
fnames = glob.glob('testImg/*.jpg', recursive=True)
print(fnames)
pirs_dataset = Pirs(fnames, transform)

print(pirs_dataset.__len__())

dataloader = DataLoader(pirs_dataset, batch_size=3,
                        shuffle=False, num_workers=4)

model_ft = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])

for data in dataloader:
    print("batch shape",data.shape)
    #fvecs = feature_extractor(data)


import torchvision.transforms as transforms
import torchvision.models as models
from skimage import io
import torch
import time
from mysql import connectDB
from PIL import Image

""" img = Image.open("test.jpg")


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


x = transform(img)
print(x)
x = x.unsqueeze(0)

model_ft = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
output = feature_extractor(x) 
print(output[0].shape)
print(len(torch.flatten(output[0])))
print(min(torch.flatten(output[0])))
print(max(torch.flatten(output[0])))
 """


""" #def get_vec():
model_ft = models.resnet18(pretrained=True)
### strip the last layer
#print(list(model_ft.children())[:-1])
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
### check this works
x = torch.randn([1,3,224,224])
print(x.shape)
output = feature_extractor(x) # output now has the features corresponding to input x
print(output.shape) """
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
        image = Image.open(self.img_list[0])
        tran_img = self.transform(image)

        return tran_img


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


fnames = connectDB(65000)
pirs_dataset = Pirs(fnames, transform)

print(pirs_dataset.__len__())

dataloader = DataLoader(pirs_dataset, batch_size=100,
                        shuffle=False, num_workers=4)

model_ft = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])

st = time.time()
with open('result/torch/fvecs.bin', 'wb') as f:
    for data in dataloader:
        fvecs = feature_extractor(data)

        fmt = f'{np.prod(fvecs.shape)}f'
        f.write(struct.pack(fmt, *(fvecs.flatten())))

print("time check saving vector", time.time()-st)
with open('result/torch/fnames.txt', 'w') as f:
        f.write('\n'.join(fnames)) 
  
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

model = models.resnet50(pretrained=True)

feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

img = Image.open("test.jpg")
x = torch.randn([1,3,224,224])
print(x.shape)

output = feature_extractor(x)
print(output.shape)
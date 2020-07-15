from tqdm import tqdm

import torch
import torchvision.models as models

from config.config import CgdConfig, ExtractorCofing
from app.model.cgd import *
from app.model.dataloader import extDataset
from config.config import ExtractorCofing

class Resnet18():
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        #self.FE = torch.nn.Sequential(*list(self.model.children())[:-1])
        print("[RESNET18] Loading Resnet18 vector extractor Model...")
    
    def featureExt(self, vecs):
        # load model & image on cuda
        self.model.cuda(), self.model.eval()

        loader = torch.utils.data.DataLoader(
            extDataset(vecs),
            batch_size=ExtractorCofing.BATCH, shuffle=False, num_workers=1
        )

        vecList = []
        # extract vecs
        with torch.no_grad():
            for _, vec in enumerate(tqdm(loader)):
                vec = vec.cuda()
                vecs = torch.nn.Sequential(*list(self.model.children())[:-1])(vec).squeeze(0)
                vecList.append(vecs)

        torch.cuda.empty_cache()

        return torch.cat(vecList)  


class ResNeXt101():
    def __init__(self):
        self.model = models.resnext101_32x8d(pretrained=True)
        print("[RESNET101_32x8d] Loading Resnet18 vector extractor Model...")
    
    def featureExt(self, vecs):
        # load model & image on cuda
        self.model.cuda(), self.model.eval()

        loader = torch.utils.data.DataLoader(
            extDataset(vecs),
            batch_size=ExtractorCofing.BATCH, shuffle=False, num_workers=1
        )

        vecList = []
        # extract vecs
        with torch.no_grad():
            for _, vec in enumerate(tqdm(loader)):
                vec = vec.cuda()
                vecs = torch.nn.Sequential(*list(self.model.children())[:-1])(vec).squeeze(0)
                vecList.append(vecs)

        torch.cuda.empty_cache()

        return torch.cat(vecList) 


class InceptionV3():
    def __init__(self):
        self.model = models.inception_v3(pretrained=True)
        print("[RESNET101_32x8d] Loading Resnet18 vector extractor Model...")
    
    def featureExt(self, vecs):
        # load model & image on cuda
        self.model.cuda(), self.model.eval()

        loader = torch.utils.data.DataLoader(
            extDataset(vecs),
            batch_size=ExtractorCofing.BATCH, shuffle=False, num_workers=1
        )

        vecList = []
        # extract vecs
        with torch.no_grad():
            for _, vec in enumerate(tqdm(loader)):
                vec = vec.cuda()
                vecs = torch.nn.Sequential(*list(self.model.children())[:-1])(vec).squeeze(0)
                vecList.append(vecs)

        torch.cuda.empty_cache()

        return torch.cat(vecList) 


class Cgd:
    def __init__(self):
        # define model params
        self.cgdName = CgdConfig.MODEL_NUM
        self.model_params = {}
        self.model_params['architecture'] = CgdConfig.ARCHITECTURE
        self.model_params['output_dim'] = CgdConfig.OUTPUT_DIM
        self.model_params['combination'] = CgdConfig.COMBINATION
        self.model_params['pretrained'] = CgdConfig.PRETRAINED
        self.model_params['classes'] = CgdConfig.CLASSES

        # define device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # define model and weight
        self.weight = torch.load(f'{CgdConfig.WEIGHTDIR}model{self.cgdName}_best.pth.tar')
        self.model =  init_network(self.model_params).to(self.device)
        self.model.load_state_dict(self.weight['state_dict'])

        print(f'[CGD{self.cgdName}] Loading CGD{self.cgdName} vector extractor Model...')
    

    def featureExt(self, inputVec):
        # load model & image on cuda
        self.model.cuda(), self.model.eval()

        inputVec = inputVec.cuda()

        # extract vecs
        extVecs = self.model(inputVec)[0].cpu().squeeze()

        torch.cuda.empty_cache()

        return extVecs
    
    def featureExt_with_dataloaer(self, inputVec):
        # load model & image on cuda
        self.model.cuda(), self.model.eval()

        loader = torch.utils.data.DataLoader(
            extDataset(inputVec),
            batch_size=ExtractorCofing.BATCH, shuffle=False, num_workers=1, pin_memory=True
        )

        vecList = []
        # extract vecs
        with torch.no_grad():
            for _, vec in enumerate(tqdm(loader)):
                vec = vec.cuda()
                vecList.append(self.model(vec)[0].cpu().squeeze())

        torch.cuda.empty_cache()

        return torch.cat(vecList, 0)

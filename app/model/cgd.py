import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from pytorchcv.model_provider import get_model as ptcv_get_model
from app.model.cirtorch.layers.pooling import MAC, SPoC, GeM

from app.model.cirtorch.layers.normalization import L2N

COMBINATION = {
    'SM'    : [SPoC, MAC],
    'MS'    : [MAC, SPoC],
    'SG'    : [SPoC, GeM],
    'GS'    : [GeM, SPoC],
    'MG'    : [MAC, GeM],
    'GM'    : [GeM, MAC],
}


class CGD(nn.Module):

    def __init__(self, features, combination, output_dim, classes, meta):
        super(CGD, self).__init__()
        self.features = nn.Sequential(*features)
        self.norm = L2N()
        self.fcr1 = nn.Linear(2048, output_dim // len(combination))
        self.fcr2 = nn.Linear(2048, output_dim // len(combination))
        self.fcc = nn.Linear(2048, classes)
        self.bn = nn.BatchNorm2d(2048)
        self.pool1, self.pool2 = combination[0](), combination[1]()
        self.meta = meta

    def forward(self, x):
        out = self.features(x) # BCWH
        # RANKING
        # First GD
        out1 = self.pool1(out)
        out1_c  = out1
        out1 = out1.permute([0, 2, 3, 1]) # BWHC
        out1 = self.fcr1(out1) # 2048 -> 768
        out1 = self.norm(out1)
        # Second GD
        out2 = self.pool2(out)
        out2 = out2.permute([0, 2, 3, 1])
        out2 = self.fcr2(out2) # 2048 -> 768
        out2 = self.norm(out2)
        # Concat Multiple GD
        outr = torch.cat((out1,out2),-1) # 768 + 768
        outr = self.norm(outr)

        # CLASSIFICATION 
        out = self.bn(out1_c)
        out = out.permute([0, 2, 3, 1])
        outc = self.fcc(out) # 2048 -> classes

        return outr, outc # BWHC


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'seresnet50')
    combination = params.get('combination', 'MG')
    pretrained = params.get('pretrained', True)
    classes = params.get('classes', 17331)
    output_dim = params.get('output_dim', 1536)

    # loading backbone network from pytorchcv
    # Github : https://github.com/osmr/imgclsmob/tree/master/pytorch
    if pretrained:
        net_in = ptcv_get_model(architecture, pretrained=True)
    else:
        net_in = ptcv_get_model(architecture, pretrained=False)

    features = list(net_in.children())[:-1]
    combination = COMBINATION[combination]

    meta = {
        'architecture': architecture,
        'outputdim': output_dim,
        'classes': classes
    }

    net = CGD(features, combination, output_dim, classes, meta)

    return net


class CGD_old(nn.Module):

    def __init__(self, features, combination, output_dim, classes, meta):
        super(CGD, self).__init__()
        self.features = nn.Sequential(*features)
        self.norm = L2N()
        self.fcr1 = nn.Linear(2048, output_dim // len(combination))
        self.fcr2 = nn.Linear(2048, output_dim // len(combination))
        self.fcc = nn.Linear(2048, classes)
        self.pool1, self.pool2 = combination[0](), combination[1]()
        self.meta = meta

    def forward(self, x):
        out = self.features(x) # BCWH
        # RANKING
        # First GD
        out1 = self.pool1(out)
        out1 = out1.permute([0, 2, 3, 1]) # BWHC
        out1 = self.fcr1(out1) # 2048 -> 768
        out1 = self.norm(out1)
        # Second GD
        out2 = self.pool2(out)
        out2 = out2.permute([0, 2, 3, 1])
        out2 = self.fcr2(out2) # 2048 -> 768
        out2 = self.norm(out2)
        # Concat Multiple GD
        outr = torch.cat((out1,out2),-1) # 768 + 768
        outr = self.norm(outr)

        # CLASSIFICATION
        out = out.permute([0, 2, 3, 1])
        outc = self.fcc(out) # 2048 -> classes

        return outr, outc # BWHC


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'seresnet50')
    combination = params.get('combination', 'MG')
    pretrained = params.get('pretrained', True)
    classes = params.get('classes', 17331)
    output_dim = params.get('output_dim', 1536)

    # loading backbone network from pytorchcv
    # Github : https://github.com/osmr/imgclsmob/tree/master/pytorch
    if pretrained:
        net_in = ptcv_get_model(architecture, pretrained=True)
    else:
        net_in = ptcv_get_model(architecture, pretrained=False)

    features = list(net_in.children())[:-1]
    combination = COMBINATION[combination]

    meta = {
        'architecture': architecture,
        'outputdim': output_dim,
        'classes': classes
    }

    net = CGD(features, combination, output_dim, classes, meta)

    return net





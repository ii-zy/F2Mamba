import torch
import torch.nn as nn
from models.DnCNN_noiseprint import make_net
import logging
import os
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from models.bdcn import BDCN
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms.functional import to_tensor, to_pil_image

class SRM(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]

        f2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]

        f3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]

        q = torch.tensor([[4.], [12.], [2.]]).unsqueeze(-1).unsqueeze(-1)
        filters = torch.tensor([[f1, f1, f1], [f2, f2, f2], [f3, f3, f3]], dtype=torch.float) / q
        self.register_buffer('filters', filters)
        self.truc = nn.Hardtanh(-2, 2)

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding='same', stride=1)
        x = self.truc(x)
        return x

class Bayar(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super().__init__()     
        self.kernel = nn.Parameter(torch.rand(self.out_channels,self.in_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)

    def bayarConstraint(self):
        self.kernel.data = self.kernel.data.div(self.kernel.data.sum(-1, keepdims=True))
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x
    
class BDCNModule(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.model = BDCN()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.device = device
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            out = self.model(x)[0]  
        return out


class LaplacianModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        edges = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edge = cv2.Laplacian(gray, cv2.CV_64F)
            edge = np.uint8(np.abs(edge))
            edge = torch.tensor(edge / 255., dtype=torch.float).unsqueeze(0)  
            edges.append(edge)
        return torch.stack(edges).to(x.device)  

class BilateralModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x): 
        edges = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
            edge = cv2.Canny(filtered, 50, 150)
            edge = torch.tensor(edge / 255., dtype=torch.float).unsqueeze(0)  
            edges.append(edge)
        return torch.stack(edges).to(x.device) 

class ForgeryAwareCluesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint', 'bayar', 'srm','bdcn', 'laplacian', 'bilateral'),
                 noiseprint_path: str = '/home/law/HDD/i_zzy/00_NEMLoc/weights/Noiseprint++.pth',
                 bdcn_weight_path: str = '/home/law/HDD/i_zzy/00_NEMLoc/weights/BDCN.pth',
                 device='cuda'):
        super().__init__()
        self.device = device
        self.mod_extract = []

        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = make_net(3, kernels=[3, ] * num_levels,features=[64, ] * (num_levels - 1) + [out_channel],bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],acts=['relu', ] * (num_levels - 1) + ['linear', ],dilats=[1, ] * num_levels,bn_momentum=0.1, padding=1)
            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights)
                dat = torch.load(np_weights, map_location=torch.device('cpu'))
                logging.info(f'Noiseprint++ weights: {np_weights}')
                self.noiseprint.load_state_dict(dat)
            self.noiseprint.eval()
            for param in self.noiseprint.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.noiseprint)
        if 'bayar' in modals:
            self.bayar = Bayar(3, 3, padding=2)
            self.mod_extract.append(self.bayar)
        if 'srm' in modals:
            self.srm = SRM()
            self.mod_extract.append(self.srm)
        if 'bdcn' in modals:
            self.bdcn = BDCN().to(device)
            assert bdcn_weight_path is not None
            self.bdcn.load_state_dict(torch.load(bdcn_weight_path, map_location=device))
            logging.info(f'BDCN weights: {bdcn_weight_path}')
            self.bdcn.eval()
            for p in self.bdcn.parameters():
                p.requires_grad = False
            self.mod_extract.append('bdcn')
        if 'laplacian' in modals:
            self.mod_extract.append('laplacian')
        if 'bilateral' in modals:
            self.mod_extract.append('bilateral')

    def set_train(self):
        if hasattr(self, 'bayar'):
            self.bayar.train()

    def set_val(self):
        if hasattr(self, 'bayar'):
            self.bayar.eval()

    def forward(self, x):
        out = []
        for mod in self.mod_extract:
            if isinstance(mod, nn.Module):
                y = mod(x)
                if y.size(1) == 1:
                    y = y.repeat(1, 3, 1, 1)
                out.append(y)
            elif mod == 'bdcn':
                with torch.no_grad():
                    bdcn_out = self.bdcn(x)[0]  
                    bdcn_out = bdcn_out.repeat(1, 3, 1, 1)
                    out.append(bdcn_out)
            elif mod == 'laplacian':
                edge_maps = [] 
                for img in x:
                    img_np = img.permute(1, 2, 0).cpu().numpy() * 255
                    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    lap = torch.tensor(np.abs(lap) / 255., dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
                    edge_maps.append(lap)
                edge_maps = torch.stack(edge_maps).to(x.device) 
                out.append(edge_maps)
            elif mod == 'bilateral':
                edge_maps = []
                for img in x:
                    img_np = img.permute(1, 2, 0).cpu().numpy() * 255
                    img_np = img_np.astype(np.uint8)
                    filtered = cv2.bilateralFilter(img_np, 9, 75, 75)
                    edge = cv2.Canny(filtered, 50, 150)
                    edge = torch.tensor(edge / 255., dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
                    edge_maps.append(edge)
                edge_maps = torch.stack(edge_maps).to(x.device)
                out.append(edge_maps)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



if __name__ == '__main__':
    modal_ext = ForgeryAwareCluesExtractor(['noiseprint', 'bayar', 'srm','bdcn', 'laplacian', 'bilateral'], 
                                    noiseprint_path='/home/law/HDD/i_zzy/00_NEMLoc/weights/Noiseprint++.pth',
                                    bdcn_weight_path='/home/law/HDD/i_zzy/00_NEMLoc/weights/BDCN.pth',device='cuda')
    
    


    

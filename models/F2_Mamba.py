import os
import time
import numpy as np
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from models.mamba_related import Linear2d,PatchMerging2D,LayerNorm2d,Permute,Mlp,gMlp,SoftmaxSpatial,mamba_init,SS2Dv0,SS2Dv2,SS2Dv3,SS2Dm0,SS2D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchsummary import summary
import yaml
from models.DnCNN_noiseprint import make_net 
import logging as logger
from models.Fine-grained_Forgery-aware_Adapter import SRM,Bayar,BDCNModule,LaplacianModule,BilateralModule,ForgeryAwareCluesExtractor
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

def get_device(cuda_idx):
    cuda_device = os.getenv("CUDA_VISIBLE_DEVICES", str(cuda_idx)) 
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")
    return device
with open('./F^2_Mamba/config_F^2Mamba.yaml', 'r') as f:
    args = yaml.safe_load(f)
device = get_device(args["cuda_idx"])
try:
    from .csms6s import selective_scan_flop_jit
except:
    from csms6s import selective_scan_flop_jit

class VSSBlock(nn.Module):
    def __init__(
            self,hidden_dim: int = 0,drop_path: float = 0,norm_layer: nn.Module = nn.LayerNorm,channel_first=False,ssm_d_state: int = 16,ssm_ratio=2.0,ssm_dt_rank: Any = "auto",ssm_act_layer=nn.SiLU,ssm_conv: int = 3,ssm_conv_bias=True,ssm_drop_rate: float = 0,ssm_init="v0",forward_type="v2",mlp_ratio=4.0,mlp_act_layer=nn.GELU,mlp_drop_rate: float = 0.0,gmlp=False,use_checkpoint: bool = False,post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)
    def _forward(self, input: torch.Tensor,low_feat: Optional[torch.Tensor] = None):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) 
        if low_feat is not None:
            x = self.low_freq_enhancer(x, low_feat)
        return x

    def forward(self, input: torch.Tensor,low_feat: Optional[torch.Tensor] = None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, low_feat)
        else:
            return self._forward(input, low_feat)
        


class VSSM(nn.Module):
    def __init__(
            self,patch_size=4,in_chans=3,num_classes=1000,depths=[2, 2, 9, 2],dims=[96, 192, 384, 768],ssm_d_state=1,ssm_ratio=1.0,ssm_dt_rank="auto",ssm_act_layer="silu",ssm_conv=3,ssm_conv_bias=False,ssm_drop_rate=0.0,ssm_init="v0",forward_type="v2_noz",
            mlp_ratio=4.0,mlp_act_layer="gelu",mlp_drop_rate=0.0,gmlp=False,drop_path_rate=0.1,patch_norm=True,
            norm_layer="LN2D", 
            downsample_version: str = "v3", 
            patchembed_version: str = "v2",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), 
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)
    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}
    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                          channel_first=False):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )
    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )
    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))
    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x
    def flops(self, shape=(3, 224, 224), verbose=True):
        supported_ops = {
            "aten::silu": None,  
            "aten::neg": None,  
            "aten::exp": None,  
            "aten::flip": None, 
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False
        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4],
                                                             align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)
class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ScoreNetwork(nn.Module):

    def __init__(self, in_ch, out_ch, mid_ch=192):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=7, stride=1, padding=3)
        self.norm = LayerNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, in_ch, Ht, Wt)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)  # (B, out_ch, Ht, Wt)
        x = x.float()      # 保证 softmax 稳定
        x = self.softmax(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def fuseforward(self, x):
        return self.act(self.conv(x))

class EarlyConv(nn.Module):
    def __init__(self, depth=3, in_channels=3, out_channels=None):
        super().__init__()
        self.depth = depth
        channels = [in_channels]
        if out_channels is None:
            out_channels = in_channels
        channels.extend([24*2**i for i in range(depth)])
        self.convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], 3, 1, 1),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU()
                )
                for i in range(depth)
            ]
        )
        self.final = nn.Conv2d(channels[-1], out_channels, 1, 1, 0)
    def forward(self, x):
        x = self.convs(x)
        x = self.final(x)
        return x

class ForgeryAwareCluesExtractor(nn.Module):
    def __init__(self, modals=['noiseprint', 'bayar', 'srm', 'bdcn', 'laplacian', 'bilateral'],
                 in_channels=[3,3,3,3,3,3], out_channels=3):
        super().__init__()
        assert len(modals) == len(in_channels)
        self.modals = modals
        self.blocks = nn.ModuleList([EarlyConv(depth=3, in_channels=in_c, out_channels=24) for in_c in in_channels])
        self.dropout = nn.Dropout(0.33)
        self.mixer = EarlyConv(depth=3, in_channels=24 * len(modals), out_channels=out_channels)
    def forward(self, x_list):
        assert len(x_list) == len(self.modals)
        m = []
        for i, blk in enumerate(self.blocks):
            m.append(blk(x_list[i]))
        x = torch.cat(m, dim=1)
        x = self.dropout(x)
        x = self.mixer(x)
        return x

class ForgeryAwareCluesExtractor(nn.Module):
    def __init__(self,
                 modals: list = ('noiseprint', 'bayar','srm','bdcn','laplacian','bilateral'),
                 noiseprint_path: str ='/home/law/HDD/i_zzy/00_NEMLoc/weights/Noiseprint++.pth',
                 bdcn_path: str = '/home/law/HDD/i_zzy/00_NEMLoc/weights/BDCN.pth'):
        super().__init__()
        self.mod_extract = nn.ModuleList()
        if 'noiseprint' in modals:
            num_levels = 17
            out_channel = 1
            self.noiseprint = make_net(3, kernels=[3, ] * num_levels,features=[64, ] * (num_levels - 1) + [out_channel],bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],acts=['relu', ] * (num_levels - 1) + ['linear', ],dilats=[1, ] * num_levels,bn_momentum=0.1, padding=1)
            if noiseprint_path:
                np_weights = noiseprint_path
                assert os.path.isfile(np_weights)
                dat = torch.load(np_weights, map_location=torch.device('cpu'))
                print(f'Noiseprint++ weights: {np_weights}')
                self.noiseprint.load_state_dict(dat)
            self.noiseprint.eval()
            for param in self.noiseprint.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.noiseprint)
        if 'bayar' in modals:
            self.bayar = BayarConv2d(3, 3, padding=2)
            self.mod_extract.append(self.bayar)
        if 'srm' in modals:
            self.srm = SRMFilter()
            self.mod_extract.append(self.srm)
        if 'bdcn' in modals:
            self.bdcn = BDCNModule(model_path='/home/law/HDD/i_zzy/00_NEMLoc/weights/BDCN.pth')
            if bdcn_path:
                assert os.path.isfile(bdcn_path)
                state = torch.load(bdcn_path, map_location='cpu')
                self.bdcn.load_state_dict(state, strict=False)
                print(f'bdcn weights: {bdcn_path}')
            self.bdcn.eval()
            for param in self.bdcn.parameters():
                param.requires_grad = False
            self.mod_extract.append(self.bdcn)
        if 'laplacian' in modals :
            self.laplacian = LaplacianModule()
            self.mod_extract.append(self.laplacian)
        if 'bilateral' in modals:
            self.bilateral = BilateralModule()
            self.mod_extract.append(self.bilateral)
    def set_train(self):
        if hasattr(self, 'bayar'):
            self.bayar.train()
    def set_val(self):
        if hasattr(self, 'bayar'):
            self.bayar.eval()
    def multi_output(self, x) -> list:
        out = []
        for modal in self.mod_extract:
            y = modal(x)
            if y.size()[-3] == 1:
                y = y.repeat((1,3, 1, 1))
            out.append(y)
        return out

class Forgery_Guided_Refinement_Decoder(nn.Module):
    def __init__(self, num_classes=2, in_channels=[96, 192, 384, 768], embedding_dim=96, dropout_ratio=0.1):
        super(Forgery_Guided_Refinement_Decoder, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels  
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim*64)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim*16)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim*4)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.conv_modals = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4+16,
            c2=embedding_dim,
            k=1,
        )
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)
    def forward(self, inputs,x_modals):
        c1, c2, c3, c4 = inputs
        x_modals=self.conv_modals(x_modals)
        n, _, h, w = c4.shape
        ps4 = nn.PixelShuffle(8)
        ps3 = nn.PixelShuffle(4)
        ps2 = nn.PixelShuffle(2)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = ps4(_c4)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = ps3(_c3)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = ps2(_c2)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1,x_modals], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x    
def crf_refine(img_np, probs_np, n_iter=5, sxy_gaussian=(3, 3), compat_gaussian=3,
               sxy_bilateral=(80, 80), srgb_bilateral=(13, 13, 13), compat_bilateral=10):
    H, W = img_np.shape[:2]
    num_classes = probs_np.shape[0]
    d = dcrf.DenseCRF2D(W, H, num_classes)
    unary = unary_from_softmax(probs_np)
    d.setUnaryEnergy(unary)
    feats_gaussian = create_pairwise_gaussian(img_np.shape[:2], sxy_gaussian)
    d.addPairwiseEnergy(feats_gaussian, compat=compat_gaussian,
                        kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats_bilateral = create_pairwise_bilateral(sdims=sxy_bilateral, schan=srgb_bilateral,
                                                img=img_np, chdim=2)
    d.addPairwiseEnergy(feats_bilateral, compat=compat_bilateral,
                        kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(n_iter)
    refined_prob = np.array(Q).reshape((num_classes, H, W))
    return refined_prob

class Encoder_Vmamba(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
        del self.classifier
        self.load_pretrained(pretrained)
    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")
    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())
        if len(self.out_indices) == 0:
            return x
        return outs


class F2Mamba_Loc(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'F2Mamba_Loc'
        self.backbone = Encoder_Vmamba()
        self.decode = Forgery_Guided_Refinement_Decoder()
        self.modalmixer = ForgeryAwareCluesExtractor()  
        self.modal_extractor = ForgeryAwareCluesExtractor(['noiseprint', 'bayar', 'srm','bdcn', 'laplacian', 'bilateral'],
                                                   r'./Noiseprint++.pth',
                                                   r'./BDCN.pth')
    def forward(self, x,use_crf=False):
        n,_,h,w=x.shape
        y=self.backbone(x)
        x_modals = F.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=False)
        x_modals = self.modal_extractor.multi_output(x_modals)
        x_modals= self.modalmixer(x_modals)
        logits = self.decode(y, x_modals) 
        prob = torch.sigmoid(logits)
        prob = F.interpolate(prob, size=(h, w), mode='bilinear', align_corners=False)
        if not use_crf:
            return prob
        refined_probs = []
        for i in range(n):
            img_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            prob_np = prob[i, 0].cpu().detach().numpy()
            prob_np_2c = np.stack([1 - prob_np, prob_np], axis=0)  
            refined_prob = crf_refine(img_np, prob_np_2c, n_iter=5)
            refined_probs.append(refined_prob[1]) 
        refined_prob_tensor = torch.tensor(np.stack(refined_probs), device=x.device).unsqueeze(1)
        return refined_prob_tensor


if __name__ == "__main__":

    model = F2Mamba_Loc().to(device)
    x = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")






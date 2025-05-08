from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .Model3D import InceptionI3d
from .fusion_modules import SumFusion3, ConcatFusion3, CMML3


class RGBEncoder(nn.Module):
    def __init__(self, config):
        super(RGBEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=3)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/wxx/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.rgbmodel = model

    def forward(self, x):
        out = self.rgbmodel(x)
        return out  # BxNx2048


class OFEncoder(nn.Module):
    def __init__(self, config):
        super(OFEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=2)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/wxx/checkpoint/flow_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.ofmodel = model

    def forward(self, x):
        out = self.ofmodel(x)
        return out  # BxNx2048


class DepthEncoder(nn.Module):
    def __init__(self, config):
        super(DepthEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=1)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        
        pretrained_dict = torch.load('/data/wxx/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.depthmodel = model

    def forward(self, x):
        out = self.depthmodel(x)
        return out  # BxNx2048


class RGBClsModel(nn.Module):
    def __init__(self, config):
        super(RGBClsModel, self).__init__()
        self.rgb_encoder = RGBEncoder(config)

        self.hidden_dim = 1024
        self.cls_r = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        rgb = x1
        rgb_feat = self.rgb_encoder(rgb)
        result_r = self.cls_r(rgb_feat)
        return result_r


class OFClsModel(nn.Module):
    def __init__(self, config):
        super(OFClsModel, self).__init__()
        self.of_encoder = OFEncoder(config)

        self.hidden_dim = 1024
        self.cls_o = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        of = x1
        of_feat = self.of_encoder(of)
        result_o = self.cls_o(of_feat)
        return result_o


class DepthClsModel(nn.Module):
    def __init__(self, config):
        super(DepthClsModel, self).__init__()
        self.depth_encoder = DepthEncoder(config)

        self.hidden_dim = 1024
        self.cls_d = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64),
            nn.Linear(64, config['setting']['num_class'])
        )

    def forward(self, x1):
        depth = x1
        depth_feat = self.depth_encoder(depth)
        result_d = self.cls_d(depth_feat)
        return result_d


class MixupClassfier(nn.Module):
    def __init__(self, args):
        super(MixupClassfier, self).__init__()
        self.rgb_encoder = RGBEncoder(args)
        self.of_encoder = OFEncoder(args)
        self.depth_encoder = DepthEncoder(args)

        self.head_m1 = nn.Linear(1024, 25)
        self.head_m2 = nn.Linear(1024, 25)
        self.head_m3 = nn.Linear(1024, 25)

        if args.mixup_method == 'single_cls':
            self.head_mixup = nn.Linear(1024, 25)
        elif args.mixup_method == 'tri_cls':
            self.head_mixup_1= nn.Linear(1024, 25)
            self.head_mixup_2= nn.Linear(1024, 25)
            self.head_mixup_3= nn.Linear(1024, 25)

    def forward(self, rgb, of, depth):
        rgb_feature = self.rgb_encoder(rgb) # (bs,1024)
        of_feature = self.of_encoder(of) # (bs, 1024)
        depth_feature = self.depth_encoder(depth) # (bs,1024)
        return rgb_feature, of_feature, depth_feature
    


class NaiveClassfier(nn.Module):
    def __init__(self, args):
        super(NaiveClassfier, self).__init__()
        self.rgb_encoder = RGBEncoder(args)
        self.of_encoder = OFEncoder(args)
        self.depth_encoder = DepthEncoder(args)
        if args.fusion_method == 'sum':
            self.fusion_module = SumFusion3(input_dim=1024, output_dim=25)
        elif args.fusion_method == 'concat':
            self.fusion_module = ConcatFusion3(input_dim = 1024*3, output_dim=25)
        elif args.fusion_method == 'weight':
            self.fusion_module = CMML3(input_dim=1024, output_dim=25)

    def forward(self, rgb, of, depth):

        rgb_feature = self.rgb_encoder(rgb) # (bs,1024)
        of_feature = self.of_encoder(of) # (bs, 1024)
        depth_feature = self.depth_encoder(depth) # (bs,1024)
        out_m1, out_m2, out_m3, out = self.fusion_module(rgb_feature, of_feature, depth_feature)
        return out_m1, out_m2, out_m3, out
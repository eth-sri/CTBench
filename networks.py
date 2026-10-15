import torch
import torch.nn as nn
from AIDomains.concrete_layers import Normalization
from AIDomains.abstract_layers import Sequential, Flatten, Linear, ReLU, Conv2d, _BatchNorm, BatchNorm2d, BatchNorm1d
import numpy as np
import math
from scipy.linalg import hadamard
from utils import fuse_BN

def fc_3layer(normalize, in_ch=3, in_dim=32, width=64, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Flatten(),
        nn.Linear(in_ch * in_dim**2, 4*width),
        nn.ReLU(),
        nn.Linear(4*width, width),
        nn.ReLU(),
        nn.Linear(width, num_class)
    )
    return model

def fc_3layer_bn(normalize, in_ch=3, in_dim=32, width=64, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Flatten(),
        nn.Linear(in_ch * in_dim**2, 4*width),
        nn.BatchNorm1d(4*width),
        nn.ReLU(),
        nn.Linear(4*width, width),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.Linear(width, num_class)
    )
    return model

def fc_5layer(normalize, in_ch=3, in_dim=32, width=64, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Flatten(),
        nn.Linear(in_ch * in_dim**2, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, num_class)
    )
    return model

def fc_5layer_bn(normalize, in_ch=3, in_dim=32, width=64, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Flatten(),
        nn.Linear(in_ch * in_dim**2, width),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.BatchNorm1d(width),
        nn.ReLU(),
        nn.Linear(width, num_class)
    )
    return model

def DLN_3layer(normalize, in_ch=3, in_dim=32, width=64, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Flatten(),
        nn.Linear(in_ch * in_dim**2, 4*width),
        nn.Linear(4*width, width),
        nn.Linear(width, num_class)
    )
    return model

def DLN_conv_3layer(normalize, in_ch=3, in_dim=32, width=64, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, num_class)
    )
    return model


def DLN_conv_3layer_bn(normalize, in_ch=3, in_dim=32, width=64, linear_size=64, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.BatchNorm2d(width),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2*width),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_3layer(normalize, in_ch=3, in_dim=32, width=16, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, num_class)
    )
    return model

def cnn_3layer_bn(normalize, in_ch=3, in_dim=32, width=16, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, num_class)
    )
    return model

def cnn_3layer_tiny(normalize, in_ch=3, in_dim=32, width=2, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * width, num_class)
    )
    return model

def cnn_4layer(normalize, in_ch=3, in_dim=32, width=16, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def ConvMed(normalize, in_ch=3, in_dim=32, width=32, linear_size=250, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 4*width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_4layer_bn(normalize, in_ch=3, in_dim=32, width=16, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_5layer(normalize, in_ch=3, in_dim=32, width=16, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def ConvMedBig(normalize, in_ch=3, in_dim=32, width=32, linear_size=250, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 4*width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def ConvMedBig_bn(normalize, in_ch=3, in_dim=32, width=32, linear_size=250, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 4*width, 4, stride=2, padding=1),
        nn.BatchNorm2d(4*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 4*width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_5layer_bn(normalize, in_ch=3, in_dim=32, width=16, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 4, stride=2, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2*width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4)**2 * 2*width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_7layer(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_7layer_bn(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model


def cnn_8layer_bn(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_9layer_bn(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_9layer(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_11layer_bn(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_11layer(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_13layer_bn(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_13layer(normalize, in_ch=3, in_dim=32, width=64, linear_size=512, num_class=10):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 2 * width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def cnn_7layer_bn_tinyimagenet(normalize, in_ch=3, in_dim=64, width=64, linear_size=512, num_class=200):
    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2*width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim//4) * (in_dim//4) * 2 * width, linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, num_class)
    )
    return model

def conv5layer_bn(normalize, in_ch=3, in_dim=32, width=64, linear_size=(512,64), num_class=10):
    linear_layers = []
    in_size = (in_dim//2) * (in_dim//2) * (width * 2)
    if linear_size is not None:
        for out_size in linear_size:
            linear_layers += [nn.Linear(in_size, out_size), nn.BatchNorm1d(out_size), nn.ReLU()]
            in_size = out_size
    linear_layers.append(nn.Linear(in_size, num_class))

    model = nn.Sequential(
        normalize,
        nn.Conv2d(in_ch, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, stride=1, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, 2 * width, 3, stride=1, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Conv2d(2 * width, width * 2, 3, stride=1, padding=1),
        nn.BatchNorm2d(width * 2),
        nn.ReLU(),
        nn.Flatten(),
        *linear_layers
    )
    return model



def get_network(model_name:str, dataset:str, device, init="default"):
    dataset = dataset.lower()
    if dataset == 'mnist':
        in_ch, in_dim, num_class = 1, 28, 10
        mean = [0.1307]
        sigma = [0.3081]
    elif dataset == 'cifar10':
        in_ch, in_dim, num_class = 3, 32, 10
        mean = [0.4914, 0.4822, 0.4465]
        sigma = [0.2023, 0.1994, 0.2010]
    elif dataset == 'tinyimagenet':
        in_ch, in_dim, num_class = 3, 64, 200
        mean = [0.4802, 0.4481, 0.3975]
        sigma = [0.2302, 0.2265, 0.2262]
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}.")

    normalize = Normalization((in_ch, in_dim, in_dim), mean, sigma)
    
    if model_name == 'cnn_7layer':
        model = cnn_7layer(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == 'cnn_7layer_bn':
        model = cnn_7layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == 'cnn_7layer_w32_bn':
        model = cnn_7layer_bn(normalize, in_ch, in_dim, width=32, num_class=num_class)
    elif model_name == 'cnn_7layer_w128_bn':
        model = cnn_7layer_bn(normalize, in_ch, in_dim, width=128, num_class=num_class)
    elif model_name == 'cnn_7layer_w256_bn':
        model = cnn_7layer_bn(normalize, in_ch, in_dim, width=256, num_class=num_class)
    elif model_name == 'cnn_7layer_w512_bn':
        model = cnn_7layer_bn(normalize, in_ch, in_dim, width=512, num_class=num_class)
    elif model_name == 'cnn_7layer_w1024_bn':
        model = cnn_7layer_bn(normalize, in_ch, in_dim, width=1024, num_class=num_class)
    elif model_name == 'cnn_7layer_bn_tinyimagenet':
        model = cnn_7layer_bn_tinyimagenet(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_8layer_bn":
        model = cnn_8layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_9layer_bn":
        model = cnn_9layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_11layer_bn":
        model = cnn_11layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_13layer_bn":
        model = cnn_13layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_9layer":
        model = cnn_9layer(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_11layer":
        model = cnn_11layer(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_13layer":
        model = cnn_13layer(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == 'cnn_5layer':
        model = cnn_5layer(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == 'cnn_5layer_bn':
        model = cnn_5layer_bn(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == 'cnn_5layer_w32_bn':
        model = cnn_5layer_bn(normalize, in_ch, in_dim, width=32, num_class=num_class)
    elif model_name == 'cnn_5layer_w64_bn':
        model = cnn_5layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == 'cnn_5layer_w128_bn':
        model = cnn_5layer_bn(normalize, in_ch, in_dim, width=128, num_class=num_class)
    elif model_name == 'cnn_5layer_w256_bn':
        model = cnn_5layer_bn(normalize, in_ch, in_dim, width=256, num_class=num_class)
    elif model_name == 'cnn_5layer_w512_bn':
        model = cnn_5layer_bn(normalize, in_ch, in_dim, width=512, num_class=num_class)
    elif model_name == 'cnn_4layer':
        model = cnn_4layer(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == 'cnn_4layer_bn':
        model = cnn_4layer_bn(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "ConvMed":
        model = ConvMed(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "ConvMedBig":
        model = ConvMedBig(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "ConvMedBig_w64":
        model = ConvMedBig(normalize, in_ch, in_dim, width=64, linear_size=500, num_class=num_class)
    elif model_name == "ConvMedBig_bn":
        model = ConvMedBig_bn(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "cnn_3layer":
        model = cnn_3layer(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "cnn_3layer_small":
        model = cnn_3layer_tiny(normalize, in_ch, in_dim, width=8, num_class=num_class)
    elif model_name == "cnn_3layer_wide":
        model = cnn_3layer(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "cnn_3layer_bn":
        model = cnn_3layer_bn(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "cnn_3layer_w100_bn":
        model = cnn_3layer_bn(normalize, in_ch, in_dim, width=100, num_class=num_class)
    elif model_name == "cnn_3layer_w300_bn":
        model = cnn_3layer_bn(normalize, in_ch, in_dim, width=300, num_class=num_class)
    elif model_name == "cnn_3layer_w600_bn":
        model = cnn_3layer_bn(normalize, in_ch, in_dim, width=600, num_class=num_class)
    elif model_name == "cnn_3layer_w900_bn":
        model = cnn_3layer_bn(normalize, in_ch, in_dim, width=900, num_class=num_class)
    elif model_name == 'fc_3layer':
        model = fc_3layer(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == 'fc_3layer_bn':
        model = fc_3layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == 'fc_5layer':
        model = fc_5layer(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == 'fc_5layer_bn':
        model = fc_5layer_bn(normalize, in_ch, in_dim, width=64, num_class=num_class)
    elif model_name == "fc_3layer_small":
        model = fc_3layer(normalize, in_ch, in_dim, width=16, num_class=num_class)
    elif model_name == "fc_3layer_wide":
        model = fc_3layer(normalize, in_ch, in_dim, width=256, num_class=num_class)
    elif model_name == "cnn_3layer_tiny":
        model = cnn_3layer_tiny(normalize, in_ch, in_dim, width=2, num_class=num_class)
    elif model_name == "cnn_3layer_small":
        model = cnn_3layer_tiny(normalize, in_ch, in_dim, width=8, num_class=num_class)
    elif model_name == "DLN_3layer":
        model = DLN_3layer(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "DLN_conv_3layer":
        model = DLN_conv_3layer(normalize, in_ch, in_dim, num_class=num_class)
    elif model_name == "DLN_conv_3layer_bn":
        model = DLN_conv_3layer_bn(normalize, in_ch, in_dim, num_class=num_class)
    else:
        raise NotImplementedError

    model = model.to(device)

    if init == "fast":
        ibp_fast_init(model) # initialization introduced in the certified training with short warmup paper
    elif init in ["default", "kaiming_uniform"]:
        model.apply(kaiming_uniform)
    elif init == "ZerO":
        ZerO_init(model)
    else:
        raise NotImplementedError(f"Unknown initialization method: {init}")

    return model

def kaiming_uniform(layer):
    if hasattr(layer, "weight") and not isinstance(layer, nn.modules.batchnorm._BatchNorm):
        torch.nn.init.kaiming_uniform_(layer.weight.data)

def get_params(model):
    weights = []
    biases = []
    for p in model.named_parameters():
        if 'weight' in p[0]:
            weights.append(p)
        elif 'bias' in p[0]:
            biases.append(p)
        else:
            print('Skipping parameter {}'.format(p[0]))
    return weights, biases

def ibp_fast_init(model):
    weights, biases = get_params(model)
    for i in range(len(weights)-1):
        if weights[i][1].ndim == 1:
            continue
        weight = weights[i][1]
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2 * math.pi / (fan_in**2))     
        std_before = weight.std().item()
        torch.nn.init.normal_(weight, mean=0, std=std)
        print(f'Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}')

def fuse_BN_wrt_Flatten(net, device, num_layer_after_flatten:int=0, remove_all=False):
    if not remove_all:
        for i, layer in enumerate(net):
            if isinstance(layer, Flatten):
                break
    else:
        i = 0
    net = fuse_BN(net, start_from=i+num_layer_after_flatten)
    return net

def add_BN_wrt_Flatten(net, device, num_layer_after_flatten:int=0):
    for i, layer in enumerate(net):
        if isinstance(layer, Flatten):
            break
    idx = int(i+num_layer_after_flatten)
    layers = list(net[:idx])
    for i, layer in enumerate(net[idx:]):
        assert not isinstance(layer, _BatchNorm), "There should not be any BN layers."
        layers.append(layer)
        if isinstance(layer, Conv2d):
            layers.append(BatchNorm2d(layer.out_channels, affine=True).to(device))
        elif isinstance(layer, Linear) and i<len(net)-idx-1:
            layers.append(BatchNorm1d(layer.out_features, affine=True).to(device))
    net = Sequential(*layers)
    net.output_dim = layers[-1].output_dim
    return net

def remove_BN_wrt_Flatten(net, device, num_layer_after_flatten:int=0, remove_all=False):
    if not remove_all:
        for i, layer in enumerate(net):
            if isinstance(layer, Flatten) or isinstance(layer, nn.Flatten):
                break
    else:
        i = 0
    idx = int(i+num_layer_after_flatten)
    layers = list(net[:idx])
    for i, layer in enumerate(net[idx:]):
        if isinstance(layer, _BatchNorm) or isinstance(layer, nn.modules.batchnorm._BatchNorm):
            continue
        layers.append(layer)
    if isinstance(net, Sequential):
        net = Sequential(*layers)
        net.output_dim = layers[-1].output_dim
    else:
        net = nn.Sequential(*layers)
    return net

# ZerO init: https://arxiv.org/pdf/2110.12661.pdf
def ZerO_init(model):
    for layer in model:
        if hasattr(layer, "bias"):
            layer.bias.data = torch.zeros_like(layer.bias.data, device=layer.bias.data.device)
        if isinstance(layer, nn.Linear):
            layer.weight.data = ZerO_Init_on_matrix(layer.weight.data)
        elif isinstance(layer, nn.Conv2d):
            k = layer.weight.shape[-1] # kernel size
            n = math.floor(k / 2)
            layer.weight.data[:, :, n, n] = ZerO_Init_on_matrix(layer.weight.data[:, :, n, n])

def ZerO_Init_on_matrix(matrix_tensor):
    # Algorithm 1 in the paper.
    m = matrix_tensor.size(0) # c_out
    n = matrix_tensor.size(1) # c_in
    device = matrix_tensor.device
    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
    
    return init_matrix.to(device)

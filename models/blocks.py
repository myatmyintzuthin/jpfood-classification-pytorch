import numpy as np
import torch.nn as nn
import torch

def activation(act: str):
    act_list = {
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'hsigmoid': nn.Hardsigmoid,
        'h-swish': nn.Hardswish,
        'silu': nn.SiLU
    }
    act = act.lower()

    if act not in act_list.keys():
        assert "activation function not supported"

    return act_list[act]

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by)*divisible_by)

class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, act='relu', groups=1) -> None:
        super(ConvBnAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        self.act = activation(act)(inplace=True)

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class ConvAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False,act='relu', groups=1) -> None:
        super(ConvAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size= kernel_size,stride=stride, padding=padding, groups=groups, bias=bias)
        self.act = activation(act)(inplace=True)

    def forward(self,x):
        return self.act(self.conv(x))

class ConvBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, groups=1) -> None:
        super(ConvBn, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size= kernel_size,stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channel)

    def forward(self,x):
        return self.bn(self.conv(x))

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias, act, groups=1) -> None:
        super(DepthWiseConv, self).__init__()

        # dw
        self.dw_conv = ConvBnAct(in_channel, in_channel, kernel_size, stride, padding, bias, act, groups)
        # pw
        self.pw_conv = ConvBn(in_channel, out_channel, 1, 1, 0, bias, 1)

    def forward(self,x):
        return self.pw_conv(self.dw_conv(x))

class SEBlock(nn.Module):
    def __init__(self, in_channel, squeeze_channel):
        super(SEBlock, self).__init__()

        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(in_channel, squeeze_channel, 1)
        self.fc2 = nn.Conv2d(squeeze_channel, in_channel, 1)
        self.relu = activation('relu')(inplace=True)
        self.hsigmoid = activation('hsigmoid')(inplace=True)

    def scale(self, input):

        x = self.globpool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hsigmoid(x)
        return x
    
    def forward(self, input):
        
        scale = self.scale(input)
        return scale * input
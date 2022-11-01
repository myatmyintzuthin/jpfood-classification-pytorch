import numpy as np
import torch.nn as nn

def activation(act: str):
    act_list = {
        'relu': nn.ReLU,
        'relu6': nn.ReLU6
    }
    act = act.lower()

    if act not in act_list.keys():
        assert "activation function not supported"

    return act_list[act]

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by)*divisible_by)

class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias, act, groups=1, num_feat=0) -> None:
        super(ConvBnAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=num_feat)
        self.relu = activation(act)(inplace=True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class ConvAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias, act, groups=1) -> None:
        super(ConvAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size= kernel_size,stride=stride, padding=padding, groups=groups, bias=bias)
        self.relu = activation(act)(inplace=True)

    def forward(self,x):
        return self.relu(self.conv(x))

class ConvBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias, groups=1, num_feat=0) -> None:
        super(ConvBn, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size= kernel_size,stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=num_feat)

    def forward(self,x):
        return self.bn(self.conv(x))

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias, act, groups=1, num_feat=0) -> None:
        super(DepthWiseConv, self).__init__()

        # dw
        self.dw_conv = ConvBnAct(in_channel, in_channel, kernel_size, stride, padding, bias, act, groups, num_feat)
        # pw
        self.pw_conv = ConvBn(in_channel, out_channel, 1, 1, 0, bias, 1, out_channel)

    def forward(self,x):
        return self.pw_conv(self.dw_conv(x))
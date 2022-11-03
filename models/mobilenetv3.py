import torch
import torch.nn as nn
import models.blocks as blocks

class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, exp_size, se, act, stride)-> None:
        super(BottleNeck, self).__init__()

        self.skip_connection = in_channel == out_channel and stride == 1
        self.squeeze_channel = blocks.make_divisible(exp_size//4, 8)
        self.block = nn.Sequential(
            blocks.ConvBnAct(in_channel, exp_size, kernel_size=1, stride=1, padding=1//2, bias=False, act=act) if exp_size != in_channel else nn.Identity(),
            blocks.ConvBnAct(exp_size, exp_size, kernel_size=kernel_size, stride=stride, act=act, groups=exp_size),
            blocks.SEBlock(exp_size, self.squeeze_channel) if se == True else nn.Identity(),
            blocks.ConvBn(exp_size, out_channel, kernel_size=1, stride=1, padding=kernel_size//2)
        )

    def forward(self, x):
        res = self.block(x)
        if self.skip_connection:
            res += x
        return res

class MobileNetV3(nn.Module):
    def __init__(self, model_cfg, num_class) -> None:
        super(MobileNetV3, self).__init__()

        self.config = model_cfg
        # first conv layer
        self.conv = blocks.ConvBnAct(in_channel=3, out_channel=16, kernel_size=3, stride=2, act='h-swish') 
        self.layers = []
        # bottle nect layer
        for c in self.config:
            kernel_size, exp_size, in_channel, out_channel, se, nl, s = c
            act = 'relu' if nl=='RE' else 'h-swish'
            self.layers.append(BottleNeck(in_channel, out_channel, kernel_size, exp_size, se, act, s))
        
        last_out_channel = self.config[-1][3]
        last_exp = self.config[-1][1]
        out = 1024 if last_exp == 576 else 1280        
        self.layers.append(
            blocks.ConvBnAct(last_out_channel, last_exp, kernel_size=1,stride=1, act='h-swish')
        )
        self.layers = nn.Sequential(*self.layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_exp, out),
            blocks.activation('h-swish')(inplace=True),
            nn.Dropout(0.8),
            nn.Linear(out, num_class)
        )

    def forward(self, x):
        x = self.conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


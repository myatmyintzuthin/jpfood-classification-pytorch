import torch.nn as nn
import models.blocks as blocks

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1,2]

        hidden_dim = int(expansion*in_channel)
        self.skip_connection = self.stride == 1 and in_channel == out_channel

        if expansion == 1:
            self.conv = blocks.DepthWiseConv(hidden_dim, out_channel, kernel_size=3, stride=stride, padding=1,bias=False, act='relu6', groups=hidden_dim,num_feat=hidden_dim)
            
        else:
            self.conv = nn.Sequential(
                #pw
                blocks.ConvBnAct(in_channel, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False, act='relu6', groups=1, num_feat=hidden_dim),
                #dw
                blocks.DepthWiseConv(hidden_dim, out_channel, kernel_size=3, stride=stride, padding=1,bias=False, act='relu6', groups=hidden_dim,num_feat=hidden_dim)
            )
        
    def forward(self, x):
        
        if self.skip_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)    

class MobileNetV2(nn.Module):
    def __init__(self, num_class, width_multi) -> None:
        super(MobileNetV2, self).__init__()

        bottle_neck = InvertedResidual
        in_channel = 32
        last_channel = 1280
        self.cfgs = [
            [1, 16, 1 , 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.last_channel = blocks.make_divisible(last_channel * width_multi) if width_multi > 1.0 else last_channel
        self.layers = [ 
                blocks.ConvBnAct(3, in_channel, kernel_size=3, stride=2, padding=1, bias=False, act='relu6', groups=1, num_feat=in_channel)
            ]

        for t,c,n,s in self.cfgs:
            out_channel = blocks.make_divisible(c * width_multi) if t > 1 else c 
            for i in range(n):
                self.layers.append(bottle_neck(in_channel, out_channel, s if i==0 else 1, t))
                in_channel = out_channel
        
        self.layers.append(
            blocks.ConvBnAct(in_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False, act='relu6', groups=1, num_feat=self.last_channel)
            )
        self.layers.append(nn.AdaptiveAvgPool2d((1,1)))

        self.layers = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.last_channel, num_class)

    def forward(self,x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


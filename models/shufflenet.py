import torch
import torch.nn as nn
import models.blocks as blocks

def channel_shuffle(x, groups: int):
    
    batchsize, num_channels, height, width = x.size()
    channels_per_groups = num_channels // groups

    x = x.view(batchsize, groups, channels_per_groups, height, width)

    x = torch.transpose(x,1,2).contiguous()

    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):

    def __init__(self, in_channel, out_channel, stride) -> None:
        super(InvertedResidual, self).__init__()

        hidden_feat = out_channel // 2
        self.stride = stride
        if self.stride != 1:
            self.branch1 = nn.Sequential(
                blocks.ConvBn(in_channel, in_channel, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=in_channel),
                blocks.ConvBnAct(in_channel, hidden_feat, kernel_size=1, stride=1, act='relu')
            )
        else:
            self.branch1 = nn.Identity()
        
        self.branch2 = nn.Sequential(
            blocks.ConvBnAct(in_channel if stride != 1 else hidden_feat, hidden_feat, kernel_size=1, stride=1, act='relu'),
            blocks.ConvBn(hidden_feat, hidden_feat, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=hidden_feat),
            blocks.ConvBnAct(hidden_feat, hidden_feat, kernel_size=1, stride=1, padding=0, bias=False, act='relu')
        )
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self,  model_cfg: dict, num_class: int, width_multi: float = 0.1) -> None:
        super(ShuffleNetV2, self).__init__()
        
        self.cfg = model_cfg[str(width_multi)]['out_channel']
        self.repeat = [4, 8, 4]

        in_channel = self.cfg[0]
        self.conv1 = blocks.ConvBnAct(3, in_channel, kernel_size=3, stride=2, padding=1, act='relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = []
        for stage in range(len(self.repeat)):
            repeat = self.repeat[stage]
            out_channel = self.cfg[stage+1]
            for i in range(repeat):
                self.blocks.append(
                    InvertedResidual(in_channel, out_channel, 2 if i == 0 else 1)
                )
            in_channel = out_channel
        
        self.blocks = nn.Sequential(*self.blocks)

        out_channel = self.cfg[-1]
        self.conv5 = blocks.ConvBnAct(in_channel, out_channel, kernel_size=1, stride=1, act='relu')
        self.globpool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Linear(out_channel, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.blocks(x)
        x = self.conv5(x)
        x = self.globpool(x)
        x = x.view(-1, self.cfg[-1])
        x = self.classifier(x)
        return x


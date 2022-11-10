import torch
import torch.nn as nn
import models.blocks as blocks
from torchvision.ops import StochasticDepth

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)

class BottleNeck(nn.Module):

    def __init__(self, channel, layer_scale, stochastic_depth_prob):
        super(BottleNeck, self).__init__()

        self.block = nn.Sequential(
        # depth-wise
        nn.Conv2d(channel, channel, kernel_size=7, padding=3, groups=channel),
        nn.GroupNorm(num_groups=1, num_channels=channel),
        Permute([0, 2, 3, 1]),
        nn.Linear(in_features=channel, out_features=4*channel),
        blocks.activation('gelu')(),
        nn.Linear(in_features=4*channel, out_features=channel),
        Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(channel,1,1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):

        result = self.layer_scale * self.block(x)
        result = self.stochastic_depth(result)     
        result += x
        return result

class ConvNeXt(nn.Module):
    def __init__(self, model_cfg, layer_scale = 1e-6, num_classes=1000) -> None:
        super(ConvNeXt, self).__init__()

        self.channels = model_cfg['channel']
        self.repeat = model_cfg['repeat']
        self.sto_depth_prob = model_cfg['sto_depth_prob']

        layers = []
        # stem
        layers.append(nn.Sequential(
            nn.Conv2d(3, self.channels[0], kernel_size=4,stride=4),
            nn.GroupNorm(num_groups=1, num_channels=self.channels[0])
        ))

        total_stage_blk = sum(self.repeat)
        stage_blk_id = 0
        # bottleneck
        for i in range(len(self.repeat)):
            for _ in range(self.repeat[i]):

                sd_prob = self.sto_depth_prob * stage_blk_id / (total_stage_blk-1)
                layers.append(BottleNeck(self.channels[i], layer_scale, sd_prob))
                stage_blk_id += 1
            if i < len(self.repeat)-1:
                # downsampling
                layers.append(
                    nn.Sequential(
                        nn.GroupNorm(num_groups=1, num_channels=self.channels[i]),
                        nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=2, stride=2)
                    )
                )
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.GroupNorm(num_groups=1,num_channels=self.channels[-1]),
            nn.Flatten(1),
            nn.Linear(self.channels[-1], num_classes)
        )        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
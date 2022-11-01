import models.blocks as blocks
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, repeat, num_classes):
        super(VGG, self).__init__()

        filters = [64, 128, 256, 512]
        self.layer1 = self._make_layers(3, filters[0], repeat[0])
        self.layer2 = self._make_layers(filters[0], filters[1], repeat[1])
        self.layer3 = self._make_layers(filters[1], filters[2], repeat[2])
        self.layer4 = self._make_layers(filters[2], filters[3], repeat[3])
        self.layer5 = self._make_layers(filters[3], filters[3], repeat[4])

        self.relu = blocks.activation('relu')
        self.linear1 = nn.Linear(7*7*512, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

    def _make_layers(self, in_channel, out_channel, repeat):

        layers = []
        layers.append(blocks.ConvAct(in_channel=in_channel,
                    out_channel=out_channel, kernel_size=3, stride=1, padding=1, bias=True, act='relu', groups=1))
        for i in range(1, repeat):
            layers.append(blocks.ConvAct(in_channel=out_channel,
                    out_channel=out_channel, kernel_size=3, stride=1, padding=1, bias=True, act='relu', groups=1))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        stage = nn.Sequential(*layers)
        return stage

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x



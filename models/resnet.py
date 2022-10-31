import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, downsample):
        super(ResidualBlock, self).__init__()
        if downsample:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
            self.identity = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
            self.identity = nn.Sequential()

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channel

    def forward(self, x):

        identity = self.identity(x)
        x = self.conv1(x)
        x = self.conv2(x)        
        x += identity
        x = self.relu(x)
        return x


class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample) -> None:
        super().__init__()

        self.downsample = downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//4,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel//4),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel//4, out_channel//4, kernel_size=3,
                      stride=2 if downsample else 1, padding=1,bias=False),
            nn.BatchNorm2d(out_channel//4),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel//4, out_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel))

        if self.downsample or in_channel != out_channel:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1,
                          stride=2 if downsample else 1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.identity = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x):

        identity = self.identity(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, residualBlk, layers, useBottleneck, num_class):
        super(ResNet, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filter = [64, 256, 512, 1024, 2048]
        else:
            filter = [64, 64, 128, 256, 512]

        self.layer1 = self._makeLayer(
            residualBlk, filter[0], filter[1], layers[0], False)
        self.layer2 = self._makeLayer(
            residualBlk, filter[1], filter[2], layers[1], True)
        self.layer3 = self._makeLayer(
            residualBlk, filter[2], filter[3], layers[2], True)
        self.layer4 = self._makeLayer(
            residualBlk, filter[3], filter[4], layers[3], True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filter[4], num_class)

    def _makeLayer(self, residualBlk, in_channel, out_channel, repeat, downsample):

        layers = []
        layers.append(residualBlk(
            in_channel, out_channel, downsample=downsample))
        for i in range(1, repeat):
            layers.append(residualBlk(
                out_channel, out_channel, downsample=False))

        stage = nn.Sequential(*layers)

        return stage

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


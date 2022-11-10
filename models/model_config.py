from torchvision import models

resnet_config = {
    '18': {'repeat': [2, 2, 2, 2], 'torch_model': models.resnet18},
    '34': {'repeat': [3, 4, 6, 3], 'torch_model': models.resnet34},
    '50': {'repeat': [3, 4, 6, 3], 'torch_model': models.resnet50},
    '101': {'repeat': [3, 4, 23, 3], 'torch_model': models.resnet101},
    '152': {'repeat': [3, 8, 36, 3], 'torch_model': models.resnet152}
}
vgg_config = {
    '11': {'repeat': [1, 1, 2, 2, 2], 'torch_model': models.vgg11},
    '13': {'repeat': [2, 2, 2, 2, 2], 'torch_model': models.vgg13},
    '16': {'repeat': [2, 2, 3, 3, 3], 'torch_model': models.vgg16},
    '19': {'repeat': [2, 2, 4, 4, 4], 'torch_model': models.vgg19}
}
mobilenet_config = {
    'v2': {'cfg': [
        #t,c,n,s = expansion, out-channel, repeat, stride
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ], 'torch_model': models.mobilenet_v2},
    'v3-small': {'cfg': [
        # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)]
        [3, 16, 16, 16, True, 'RE', 2],
        [3, 72, 16, 24, False, 'RE', 2],
        [3, 88, 24, 24, False, 'RE', 1],
        [5, 96, 24, 40, True, 'HE', 2],
        [5, 240, 40, 40, True, 'HE', 1],
        [5, 240, 40, 40, True, 'HE', 1],
        [5, 120, 40, 48, True, 'HE', 1],
        [5, 144, 48, 48, True, 'HE', 1],
        [5, 288, 48, 96, True, 'HE', 2],
        [5, 576, 96, 96, True, 'HE', 1],
        [5, 576, 96, 96, True, 'HE', 1]
    ], 'torch_model': models.mobilenet_v3_small},
    'v3-large': {'cfg': [
        [3, 16, 16, 16, False, 'RE', 1],
        [3, 64, 16, 24, False, 'RE', 2],
        [3, 72, 24, 24, False, 'RE', 1],
        [5, 72, 24, 40, True, 'RE', 2],
        [5, 120, 40, 40, True, 'RE', 1],
        [5, 120, 40, 40, True, 'RE', 1],
        [3, 240, 40, 80, False, 'HE', 2],
        [3, 200, 80, 80, False, 'HE', 1],
        [3, 184, 80, 80, False, 'HE', 1],
        [3, 184, 80, 80, False, 'HE', 1],
        [3, 480, 80, 112, True, 'HE', 1],
        [3, 672, 112, 112, True, 'HE', 1],
        [5, 672, 112, 160, True, 'HE', 2],
        [5, 960, 160, 160, True, 'HE', 1],
        [5, 960, 160, 160, True, 'HE', 1]
    ], 'torch_model': models.mobilenet_v3_large}
}
shufflenet_config = {
    'v2': {
        '0.5': {'out_channel': [24, 48, 96, 192, 1024], 'torch_model': models.shufflenet_v2_x0_5},
        '1.0': {'out_channel': [24, 116, 232, 464, 1024], 'torch_model': models.shufflenet_v2_x1_0},
    }
}
convNeXt_config = {
    'tiny': {'channel': [96, 192, 384, 768], 'repeat': [3, 3, 9, 3], 'sto_depth_prob': 0.1,
            'torch_model': models.convnext_tiny},
    'small': {'channel': [96, 192, 384, 768], 'repeat': [3, 3, 27, 3], 'sto_depth_prob': 0.4,
            'torch_model': models.convnext_small},
    'base': {'channel': [128, 256, 512, 1024], 'repeat': [3, 3, 27, 3], 'sto_depth_prob': 0.5,
            'torch_model': models.convnext_base},
    'large': {'channel': [192, 384, 768, 1536], 'repeat': [3, 3, 27, 3], 'sto_depth_prob': 0.1,
            'torch_model': models.convnext_large}           
}

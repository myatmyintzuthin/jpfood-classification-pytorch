from torchvision import models

resnet_config = {
        '18': {'repeat':[2, 2, 2, 2], 'torch_model': models.resnet18},
        '34': {'repeat':[3, 4, 6, 3], 'torch_model': models.resnet34},
        '50': {'repeat':[3, 4, 6, 3], 'torch_model': models.resnet50},
        '101': {'repeat':[3, 4, 23, 3], 'torch_model': models.resnet101},
        '152': {'repeat':[3, 8, 36, 3], 'torch_model': models.resnet152}
    }
vgg_config = {
        '11': {'repeat':[1, 1, 2, 2, 2],'torch_model': models.vgg11},
        '13': {'repeat':[2, 2, 2, 2, 2],'torch_model': models.vgg13},
        '16': {'repeat':[2, 2, 3, 3, 3],'torch_model': models.vgg16},
        '19': {'repeat':[2, 2, 4, 4, 4],'torch_model': models.vgg19}
    }
mobilenet_config = {
        'v2': {'torch_model': models.mobilenet_v2}
    }
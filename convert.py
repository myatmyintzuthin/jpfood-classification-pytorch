import os
import argparse
import torch

from models.model_config import vgg_config, resnet_config, mobilenet_config, shufflenet_config, convNeXt_config
from models.resnet import ResNet, ResidualBlock, ResBottleneckBlock
from models.vgg import VGG
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3
from models.shufflenet import ShuffleNetV2
from models.convnext import ConvNeXt
from torchinfo import summary

class ConvertModel:
    def __init__(self, model: str, variant: str, width_multi: float, num_class: str) -> None:
        self.model = model
        self.variant = variant
        self.width_mutli = width_multi
        self.num_class = num_class

    def load_model(self):
        '''
        Load different DL models based on input
        '''
        if self.model.lower() not in ['vgg','resnet','mobilenet','shufflenet']:
            assert "model not supported"

        if self.model == 'vgg':
            model = VGG(vgg_config[str(self.variant)]['repeat'], self.num_class)
        if self.model == 'resnet':
            if self.variant == '18' or self.variant == '34':
                model = ResNet(ResidualBlock, resnet_config[str(self.variant)]['repeat'], useBottleneck=False, num_class=self.num_class)
            else:
                model = ResNet(ResBottleneckBlock, resnet_config[str(self.variant)]['repeat'], useBottleneck=True, num_class=self.num_class)
        if self.model == 'mobilenet':
            if self.variant == 'v2':
                model = MobileNetV2(mobilenet_config[str(self.variant)]['cfg'], self.num_class, self.width_mutli)
            else:
                model = MobileNetV3(mobilenet_config[str(self.variant)]['cfg'], self.num_class)
        if self.model == 'shufflenet':
            if self.variant == 'v2':
                model = ShuffleNetV2(shufflenet_config[str(self.variant)], self.num_class, self.width_mutli)
        if self.model == 'convnext':
                model = ConvNeXt(convNeXt_config[str(self.variant)],self.num_class)
        return model

    def initialize_weights(self):
        '''
        Apply pretrained weight from torchvision.models to custom built models
        '''
        model_name = f"{self.model}{self.variant}"
        my_model = self.load_model()
        my_state_dict = my_model.state_dict()

        # print(len(my_state_dict.keys()))
        # for i in my_state_dict:
        #     print(i)
        summary(my_model, input_size=(8, 3, 224, 224))

        if self.model == 'vgg':
            pretrained_model = vgg_config[str(self.variant)]['torch_model'](weights = True)
        if self.model == 'resnet':
            pretrained_model = resnet_config[str(self.variant)]['torch_model'](weights = True)
        if self.model == 'mobilenet':
            pretrained_model = mobilenet_config[str(self.variant)]['torch_model'](weights = True)
        if self.model == 'shufflenet':
            pretrained_model = shufflenet_config[str(self.variant)][str(self.width_mutli)]['torch_model'](weights = True)
        if self.model == 'convnext':
            pretrained_model = convNeXt_config[str(self.variant)]['torch_model'](weights = True)

        # summary(pretrained_model, input_size=(8, 3, 224, 224))

        pretrained_state_dict = pretrained_model.state_dict()
        # print(len(pretrained_state_dict.keys()))
        # for i in pretrained_state_dict:
        #     print(i)
        for my, pre in zip(my_state_dict.keys(), pretrained_state_dict.keys()):
            my_state_dict[my] = pretrained_state_dict[pre]

        my_model.load_state_dict(my_state_dict)

        # save model
        save_folder = 'pretrained_weights'
        save_path = os.path.join(save_folder, model_name+".pt")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(my_model.state_dict(), save_path)

        print(f'{self.model}_{self.variant} Pretrained weight is saved in {save_folder}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model name', required=True,)
    parser.add_argument('--variant', type=str,
                        help='variant name', required=True)
    parser.add_argument('--width_multi', type=float,
                        help='width multiplication ratio for mobilenet', default=1.0)
    parser.add_argument('--num_class', type=int,
                        help='number of class', default=1000)

    opt = parser.parse_args()

    prepare_model = ConvertModel(opt.model, opt.variant, opt.width_multi, opt.num_class)
    prepare_model.initialize_weights()

import os
import argparse
import torch

from models.model_config import vgg_config, resnet_config, mobilenet_config
from models.resnet import ResNet, ResidualBlock, ResBottleneckBlock
from models.vgg import VGG
from models.mobilenetv2 import MobileNetV2

class convert_model():
    def __init__(self, model: str, variant: str, width_multi: float, num_class: str) -> None:
        self.model = model
        self.variant = variant
        self.width_mutli = width_multi
        self.num_class = num_class

    def load_model(self):
        '''
        Load different DL models based on input
        '''
        if self.model == 'vgg':
            model = VGG(vgg_config[str(self.variant)]['repeat'], self.num_class)
        if self.model == 'resnet':
            if self.variant == '18' or self.variant == '34':
                model = ResNet(ResidualBlock, resnet_config[str(self.variant)]['repeat'], useBottleneck=False, num_class=self.num_class)
            else:
                model = ResNet(ResBottleneckBlock, resnet_config[str(self.variant)]['repeat'], useBottleneck=True, num_class=self.num_class)
        if self.model == 'mobilenet':
            if self.variant == 'v2':
                model = MobileNetV2(self.num_class, self.width_mutli)
        return model

    def initialize_weights(self):
        '''
        Apply pretrained weight from torchvision.models to custom built models
        '''
        model_name = f"{self.model}{self.variant}"
        my_model = self.load_model()
        my_state_dict = my_model.state_dict()

        if self.model == 'vgg':
            pretrained_model = vgg_config[str(self.variant)]['torch_model'](pretrained = True)
        if self.model == 'resnet':
            pretrained_model = resnet_config[str(self.variant)]['torch_model'](pretrained = True)
        if self.model == 'mobilenet':
            pretrained_model = mobilenet_config[str(self.variant)]['torch_model'](pretrained = True)

        pretrained_state_dict = pretrained_model.state_dict()

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

    prepare_model = convert_model(opt.model, opt.variant, opt.width_multi, opt.num_class)
    prepare_model.initialize_weights()

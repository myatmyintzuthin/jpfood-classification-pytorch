import os
import argparse
import torch
from glob import glob
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

import utils.utils as utils
import convert

class Inference():
    def __init__(self, opt) -> None:

        self.config = utils.yaml_parser(opt.cfg)
        self.image_path = opt.image_path
        self.viz = opt.viz
        self.class_name = self.config['dataset']['class_name']
        self.model_name = self.config['model']['name']
        self.variant = self.config['model']['variant']
        self.width_multi = self.config['model']['width_multi']
        self.model_path = self.config['test']['model_path']

        self.exp_dir = os.path.split(self.model_path)[0]
        self.test_log = utils.setup_logger(self.exp_dir+'/logs/test.log')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        model = self.prepare_model()
        self.inference(model)

    def prepare_model(self):
        
        choose_model = convert.convert_model(
            self.model_name, self.variant, self.width_multi, len(self.class_name))
        model = choose_model.load_model()
        ckpt = torch.load(self.model_path)
        model = utils.load_ckpt(model, ckpt)
        model = model.to(self.device)
        return model

    def inference(self, model):

        model.eval()
        image_files = glob(os.path.join(self.image_path, '*.jpg'))

        row, column = int(len(image_files)/2), int(len(image_files)/2)
        fig = plt.figure(figsize=(10,7))
        for index, image_file in enumerate(image_files):
            image = Image.open(image_file)

            # apply transfoms
            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            image_transformed = image_transform(image)

            with torch.inference_mode():
                image_transformed = image_transformed.unsqueeze_(dim=0)
                image_transformed = image_transformed.to(self.device)
                prediction = model(image_transformed)

            predicted_prob = torch.softmax(prediction, dim=1)
            predicted_class = torch.argmax(predicted_prob, dim=1)
            pred_class_name = self.class_name[predicted_class.cpu()]

            prediction_result = f"Label: {os.path.basename(image_file).split('.')[0]} | Pred: {pred_class_name} | Prob: {predicted_prob.max().cpu():3f}"
            self.test_log.info(prediction_result)
        
            fig.add_subplot(row, column, index+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"{pred_class_name} | {predicted_prob.max().cpu():3f}")

        if opt.viz:
            plt.savefig(os.path.join(self.exp_dir,'inference_result.png'))
            plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--image_path', type=str,
                        help='Path to test image', required=True)
    parser.add_argument('--viz', action='store_true',default=False, help='To visualize the inference results')
    
    opt = parser.parse_args()

    inference = Inference(opt)
    inference.run()

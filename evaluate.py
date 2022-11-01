import os
import torch
import utils.utils as utils
import metrics.metrices as metrices
import convert


class Evaluation:
    def __init__(self, model_path, test_loader, config) -> None:

        self.model_path = model_path
        self.test_loader = test_loader

        self.class_name = config['dataset']['class_name']
        self.model_name = config['model']['name']
        self.variant = config['model']['variant']
        self.width_multi = config['model']['width_multi']

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        ''' run process
        '''
        model = self.prepare_model()
        classification_report = self.evaluate(model)
        return classification_report

    def prepare_model(self):
        ''' load model
        '''
        choose_model = convert.ConvertModel(
            self.model_name, self.variant, self.width_multi, len(self.class_name))
        model = choose_model.load_model()
        ckpt = torch.load(self.model_path)
        model = utils.load_ckpt(model, ckpt)
        model = model.to(self.device)
        return model

    def evaluate(self, model):
        ''' model evaluation
        '''
        model.eval()
        actual_label, pred_label = [],[]
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.test_loader):

                X, y = X.to(self.device), y.to(self.device)

                prediction = model(X)

                pred_prob = torch.softmax(prediction, dim=1)
                predicted = torch.argmax(pred_prob, dim=1)

                actual_label.extend(y.cpu().numpy().tolist())
                pred_label.extend(predicted.cpu().numpy().tolist())

        matrix_save_path = os.path.join(os.path.split(
            self.model_path)[0], 'confusion_matrix.png')
        confusion_matrix = metrices.plot_confusion(
            actual_label, pred_label, self.class_name, save_path=matrix_save_path, save=True)
        classification_report = metrices.classification_report(
            confusion_matrix, self.class_name)

        return classification_report

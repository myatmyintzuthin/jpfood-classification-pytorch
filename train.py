import os
import torch
import torch.nn as nn
from torchinfo import summary
import argparse
from pathlib import Path
from timeit import default_timer as timer

import evaluate
import convert
import dataset.dataset as dataset
from metrics.metrices import classification_report
import utils.utils as utils
import core.trainer as trainer

torch.manual_seed(42)
torch.cuda.manual_seed(42)


class Training:
    def __init__(self, opt) -> None:
        self.config = utils.yaml_parser(opt.cfg)
        self.eval_opt = opt.eval
        self.eval_model = opt.eval_model
        
        self.dataset_cfg = self.config['dataset']
        model_cfg = self.config['model']
        train_cfg = self.config['train']

        self.num_class = len(self.dataset_cfg['class_name'])

        self.model_name = model_cfg['name']
        self.variant = model_cfg['variant']
        self.width_multi = model_cfg['width_multi']

        self.model_path = train_cfg['model_path']
        self.pretrained_path = train_cfg['pretrained_path']
        self.BATCH_SIZE = train_cfg['batch_size']
        self.epochs = train_cfg['epochs']
        self.num_worker = train_cfg['num_worker']
        self.learning_rate = train_cfg['lr']
        self.save_dir = train_cfg['save_dir']

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        ''' run process 
            train -> evaluation
        '''
        if not self.eval_opt:
            
            exp_name = self.model_name+self.variant
            self.exp_dir = utils.create_experiment(self.save_dir, exp_name)
            self.save_model = os.path.join(self.exp_dir, self.model_path)
            self.train_log = utils.setup_logger(self.exp_dir+'/logs/train.log')
        
            dataloader = self.dataloader(self.dataset_cfg, self.train_log)
            model = self.prepare_model()
            self.train(model, dataloader)
            self.eval(self.save_model, dataloader, self.train_log)
        else:
            exp_dir = os.path.split(self.eval_model)[0]
            eval_log = utils.setup_logger(exp_dir+'/logs/eval.log')
            dataloader = self.dataloader(self.dataset_cfg, eval_log)

            self.eval(self.eval_model, dataloader, eval_log)

    def dataloader(self, cfg, log):
        ''' data preparation
        '''
        data_path = Path(cfg['root'])
        image_path = data_path/cfg['dataset_name']
        dataloader = dataset.CustomDataloader(
            data_path, image_path, self.BATCH_SIZE, log, self.num_worker, shuffle=True)
        return dataloader

    def prepare_model(self):
        ''' load model
        '''
        choose_model = convert.ConvertModel(
            self.model_name, self.variant, self.width_multi, self.num_class)
        model = choose_model.load_model()

        ckpt = torch.load(self.pretrained_path)
        model = utils.load_ckpt(model, ckpt)

        model = model.to(self.device)
        summary(model, input_size=(self.BATCH_SIZE, 3, 224, 224))
        return model

    def train(self, model, dataloader):
        ''' model training
        '''
        # dataloader
        train_loader, valid_loader = dataloader.train_dataloader(), dataloader.valid_dataloader()
        # lost function
        loss_fn = nn.CrossEntropyLoss()
        # optimizer
        optimizer = torch.optim.SGD(params=model.parameters(
        ), lr=self.learning_rate, weight_decay=0.001, momentum=0.9, nesterov=True)
        # scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.2)
        self.train_log.info("Training Starts...")
        start_time = timer()
        model_results = trainer.train(
            model=model, train_dataloader=train_loader, test_dataloader=valid_loader, optimizer=optimizer, loss_fn=loss_fn,
            scheduler=scheduler, epochs=self.epochs, log=self.train_log, device=self.device)

        end_time = timer()
        self.train_log.info(
            f'Total training time: {end_time-start_time:.3f} seconds')

        # save train history
        save_history_path = os.path.join(self.exp_dir, 'training_history.png')
        utils.plot_curves(model_results, save_history_path)

        # save model
        torch.save(model.state_dict(), self.save_model)

    def eval(self, save_model, dataloader, log):
        ''' model evaluation
        '''
        log.info("Evaluation Starts....")
        test_loader = dataloader.test_dataloader()
        evaluation = evaluate.Evaluation(save_model, test_loader, self.config)
        classification_report = evaluation.run()
        log.info(classification_report)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--eval', action='store_true',
                        default=False, help='To only evaluate')
    parser.add_argument('--eval_model', type=str, help='Model path to evaluate')
    
    opt = parser.parse_args()

    train = Training(opt)
    train.run()

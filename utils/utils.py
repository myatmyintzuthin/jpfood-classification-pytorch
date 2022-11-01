import yaml
import os
import logging
from rich.logging import RichHandler
import matplotlib.pyplot as plt
from datetime import datetime

def setup_logger(log_name):
    '''
    setup rich logger
    '''
    logger = logging.getLogger(__name__)

    rh = RichHandler()
    fh = logging.FileHandler(filename=log_name, mode='w+')

    logger.setLevel(logging.INFO)
    logger.addHandler(rh)
    logger.addHandler(fh)

    return logger

def yaml_parser(yaml_file):
    '''
    yaml file praser
    Args: 
        ymal file
    Return: 
        prased dictionary
    '''
    with open(yaml_file, 'r') as stream:
        try:
            prased_dict = yaml.safe_load(stream)
            return prased_dict
        except yaml.YAMLError as exc:
            print(exc)

def create_experiment(save_dir:str, model_name: str):
    '''
    Create experiment folder for each model training.
    '''
    time = datetime.now()
    today= time.strftime('%Y%m%d%H%M')

    exp_dir = os.path.join(save_dir,f"{model_name}_{str(today)}")

    if not os.path.exists(exp_dir):
        os.makedirs(os.path.join(exp_dir,'logs'))
    
    return exp_dir

def plot_curves(results, save_path):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
            "train_acc": [...],
            "test_loss": [...],
            "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(save_path)

def load_ckpt(model, ckpt):
    '''
    load trained model 
    '''
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            continue
        load_dict[key_model] = v_ckpt
    model.load_state_dict(load_dict, strict=False)
    return model
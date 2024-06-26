import argparse, sys
import collections
import torch
import numpy as np

import model.loss as module_loss
from utils import prepare_device
import model.model as module_arch
import model.metric as module_metric
from parse_config import ConfigParser
import dataloader.loader as module_data
import trainer.trainer as module_trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    logger.info(f"SEED: {SEED}")
    
    # setup data_loader instances
    data_loader = config.init_obj('dataloader', module_data)
    
    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids)>1 and torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss, metrics and trainer
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    Trainer = getattr(module_trainer, config['trainer']["type"])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model, 
        criterion, 
        metrics, 
        optimizer,
        config=config, 
        device=device,
        lr_scheduler=lr_scheduler,
        data_loader=data_loader)

    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Argument for NNER model')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    
    config = ConfigParser.from_args(args, options)
    main(config)



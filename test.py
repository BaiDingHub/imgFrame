import dataLoader as module_loader
import model as module_model
import trainer as module_trainer
import metric as module_metric
from config import Config
from utils import *


import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    config = Config()
    dataset = getattr(module_loader,config.CONFIG['dataset_name']+'TestSet')(**getattr(config,config.CONFIG['dataset_name']))    
    dataLoader = DataLoader(dataset,batch_size = config.ARG['batch_size'],shuffle=False,num_workers=1)
    metrics = [getattr(module_metric,metric) for metric in config.CONFIG['metrics']]

    model = getattr(module_model,config.CONFIG['model_name'])()
    criterion = None

    optimizer = None

    trainer = getattr(module_trainer,config.CONFIG['model_name']+"Trainer")(model,dataLoader,criterion,
                                                                            optimizer,metrics,config)
    print('start test')
    trainer.test()


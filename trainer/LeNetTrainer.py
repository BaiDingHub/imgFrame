from trainer.BaseTrainer import *

import torch
import torch.nn as nn
from tqdm import tqdm

class LeNetTrainer(BaseTrainer):
    """[LeNet+Mnist]
    """
    def __init__(self,model,dataLoader,criterion,optimizer,metrics,config):
       super(LeNetTrainer,self).__init__(model,dataLoader,criterion,optimizer,metrics,config)


    def _train_epoch(self,epoch):

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        trainloader_t = tqdm(self.dataLoader,ncols=100)
        
        trainloader_t.set_description("Train Epoch: {}|{}  Batch size: {}  LR : {:.4}".format
                                      (epoch,self.EPOCH,self.dataLoader.batch_size,self.optimizer.param_groups[0]['lr']))
        
        for idx,(x,y) in enumerate(trainloader_t):
            if self.use_gpu:
                x = x.cuda(self.device_ids[0])
                y = y.cuda(self.device_ids[0])
            x = x.float()
            y = y.long()
            self.optimizer.zero_grad()

            logits = self.model(x)

            loss = self.criterion(logits,y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(logits.cpu().detach().numpy(),y.cpu().detach().numpy())
        
        log = {
            'loss' : total_loss /self.lenEpoch,
        }
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]/self.lenEpoch

        return log


    def _test_epoch(self):
        total_metrics = np.zeros(len(self.metrics))

        test_loader_t = tqdm(self.dataLoader,ncols=100)
        test_loader_t.set_description("Batch size: {}".format(self.dataLoader.batch_size))

        for idx,(x,y) in enumerate(test_loader_t):
            if self.use_gpu:
                x = x.cuda(self.device_ids[0])
                y = y.cuda(self.device_ids[0])
            x = x.float()
            y = y.long()
            logits = self.model(x)
            total_metrics += self._eval_metrics(logits.cpu().detach().numpy(),y.cpu().detach().numpy())
        
        log = {}
        for i,metric in enumerate(self.metrics):
            log[metric.__name__] = total_metrics[i]/self.lenEpoch

        return log


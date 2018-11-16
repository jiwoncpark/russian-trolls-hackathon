import sys
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed

from os import makedirs
from os.path import exists
from utils.dump import Dump
from net.net_constructor import get_net
from utils.lr_scheduler import MultiStepLR
from utils.average_meter import AverageMeter
from utils.check_point import save_check_point
from loader.loader_constructor import get_loader

class Experiment:
    
    def __init__(self, opts):
        
        for key, value in opts.items():
            setattr(self, key, value)
        
        if not exists(self.training_results_path):
            makedirs(self.training_results_path)
        
        self.set_seed()
        
        # create loaders
        self.TEXT, self.vocab_size, self.word_embeddings, \
        self.train_loader, self.test_loader = get_loader(self)
        
        # initialize network
        self.model = get_net(self)
        self.model.to(self.device)
        
        # criterion to optimize
        func = getattr(nn, self.crit)
        self.criterion = func()
        
        # optimizer
        func = getattr(optim, self.optim)
        opt_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = func(opt_params,
                              lr=self.lr,
                              weight_decay=self.weight_decay,
                              **self.optim_kwargs)
        
        # scheduler to change learning rate
        self.milestones = [int(self.epochs*self.milestones_perc[i]) for i in range(len(self.milestones_perc))]
        self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=self.milestones,
                                        gamma=self.gamma)
    
    
    def run(self, stats_meter, stats_no_meter):
        
        self.set_seed()
        
        for epoch in range(1, self.epochs + 1):
            
            self.lr_scheduler.step()
            
            # csv file for dumping results
            results = Dump(self.training_results_path+'/'+self.train_dump_file)
            
            # train epoch
            results = self.run_epoch("train",
                                       epoch,
                                       self.train_loader,
                                       stats_meter,
                                       stats_no_meter,
                                       results)  
            
            # test epoch
            results = self.run_epoch("test",
                                       epoch,
                                       self.test_loader,
                                       stats_meter,
                                       stats_no_meter,
                                       results)
            
            # save results to csv
            results.save()
            results.to_csv()
            
            # save model
            save_check_point(self, self.model)
            
            
    def run_epoch(self,
                  phase,
                  epoch,
                  loader,
                  stats_meter,
                  stats_no_meter,
                  results):
        
        # keep track of how long data loading and processing takes
        batch_time  = AverageMeter()
        data_time   = AverageMeter()
        
        # other meteres (top1, top5, loss, etc.)
        meters = {}
        for name, func in stats_meter.items():
            meters[name] = AverageMeter()
        
        # chnage model to train or eval mode
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise Exception('Phase must be train or test!')
        
        t = time.time()
        
        # iterate over all batches
        for iter, batch in enumerate(loader, 1):
            
            data_time.update(time.time() - t)
    
            # batch input and target output
            input   = batch[0]
            target  = batch[1]
            
            # transfer data to gpu
            input = input.to(self.device)
            target = target.to(self.device)
            
            torch.set_grad_enabled(phase == 'train')
            
            # forward pass
            est = self.model(input)
            
            # compute loss
            loss = self.criterion(est, target)
            
            # backward pass
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # update meters (top1, top5, loss, etc.)
            for name, func in stats_meter.items():
                meters[name].update(func(locals()), input.data.shape[0])
                
            batch_time.update(time.time() - t)
            
            # print to console progress
            output = '{}\t'                                                 \
                     'Network: {}\t'                                        \
                     'Epoch: [{}/{}][{}/{}]\t'                              \
                     'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  \
                     'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'    \
                     .format(phase.capitalize(),
                             self.net,
                             epoch,
                             self.epochs,
                             iter,
                             len(loader),
                             batch_time=batch_time,
                             data_time=data_time)

            for name, meter in meters.items(): 
                output = output + '{}: {meter.val:.4f} ({meter.avg:.4f})\t' \
                                  .format(name, meter=meter)
            
            print(output)
            sys.stdout.flush()
            
            if iter == len(loader):
                
                # save the following into the csv
                stats = {'phase'             : phase,
                         'epoch'             : epoch,
                         'iter'              : iter,
                         'iters'             : len(loader),
                         'iter_batch_time'   : batch_time.val,
                         'avg_batch_time'    : batch_time.avg,
                         'iter_data_time'    : data_time.val,
                         'avg_data_time'     : data_time.avg}
                
                # meters that will be saved into the csv
                for name, meter in meters.items():
                    stats['iter_'+name]     = meter.val
                    stats['avg_'+name]      = meter.avg
                    stats['sum_'+name]      = meter.sum
                    stats['count_'+name]    = meter.count
                
                # stats that have no meters but will be saved into the csv
                for name, func in stats_no_meter.items():
                    stats[name] = func(locals())
                
                # save all fields in "self" into the csv
                # these include (almost) all the fields in train.py
                results.append(dict(self.__getstate__(), **stats))
            
            t = time.time()
            
        return results
        
        
    def set_seed(self):
        # set the random seed of all devices for reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # attributes that should not be saved into the csv
        attributes = [
                      'TEXT',
                      'vocab_size',
                      'word_embeddings',
                      'train_loader',
                      'test_loader',
                      'model',
                      'criterion',
                      'optimizer',
                      'optim_kwargs',
                      'lr_scheduler',
                      'device',
                      ]
        
        # remove those attributes from the state
        for attr in attributes:
            if hasattr(self, attr):
                del state[attr]
        
        return state
        
        
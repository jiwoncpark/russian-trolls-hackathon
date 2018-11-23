import sys
import time
import torch
import random
import numpy as np
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
        self.TEXT, self.vocab_size, self.word_embeddings, self.train_loader, self.test_loader = get_loader(self)
        
        # initialize network
        self.model = get_net(self)
        self.model.to(self.device)
        
        # Metadata input + 3-tuple output
        
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
    
    
    def run(self, stats_meter, stats_no_meter, test_run):
        
        if not test_run:
            # Normal case
            self.set_seed()

            best_total_loss = np.infty
            best_loss_1 = np.infty
            best_loss_2 = np.infty
            best_loss_3 = np.infty

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
                results, epoch_test_loss, epoch_test_loss1, epoch_test_loss2, epoch_test_loss3 = self.run_epoch("test",
                                           epoch,
                                           self.test_loader,
                                           stats_meter,
                                           stats_no_meter,
                                           results)

                # save results to csv
                results.save()
                results.to_csv()

                # Update the checkpoint only when better test loss is found.
                # We have 4 loss, (total, loss_1, loss_2, loss_3) and we save 4 models for each hyperparameter

                if (best_total_loss > epoch_test_loss):
                    best_total_loss = epoch_test_loss
                    print('Better model for total loss is found and saved! The loss is {}'.format(best_total_loss))
                    save_check_point(self, self.model, 'total_loss')

                if (best_loss_1 > epoch_test_loss1):
                    best_loss_1 = epoch_test_loss1
                    print('Better model for loss 1 is found and saved! The loss is {}'.format(best_loss_1))
                    save_check_point(self, self.model, 'loss_1')

                if (best_loss_2 > epoch_test_loss2):
                    best_loss_2 = epoch_test_loss2
                    print('Better model for loss 2 is found and saved! The loss is {}'.format(best_loss_2))
                    save_check_point(self, self.model, 'loss_2')

                if (best_loss_3 > epoch_test_loss3):
                    best_loss_3 = epoch_test_loss3
                    print('Better model for loss 3 is found and saved! The loss is {}'.format(best_loss_3))
                    save_check_point(self, self.model, 'loss_3')

            return (best_total_loss, best_loss_1, best_loss_2, best_loss_3)
        else:
            # Test run case
            pass
        
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
            # if self.loader == 'basic_meta_data':
            input_text = batch[0] # tweets
            input_meta = batch[1] # metadata
            target = batch[2]
            
            # transfer data to gpu
            input_text = input_text.to(self.device)
            input_meta = input_meta.to(self.device)
            target = target.to(self.device)
            
            torch.set_grad_enabled(phase == 'train')
            
            est = self.model(input_text=input_text, input_meta=input_meta)

            loss = 0.0
            three_losses = []
            three_baseline_losses = []
                       
            for output_idx in range(3):
                flavor_est = est[:, output_idx]
                flavor_target = target[:, output_idx]
                flavor_loss = self.criterion(flavor_est, flavor_target)
                three_losses.append(flavor_loss)
                loss += flavor_loss/3.0
                # Baseline loss
                baseline_flavor_est = input_meta[:, output_idx + 6]
                baseline_flavor_loss = self.criterion(baseline_flavor_est, flavor_target)
                three_baseline_losses.append(baseline_flavor_loss)
            
            loss1 = three_losses[0]
            loss2 = three_losses[1]
            loss3 = three_losses[2]
            baseline1 = three_baseline_losses[0]
            baseline2 = three_baseline_losses[1]
            baseline3 = three_baseline_losses[2]
            
            ratios = []
            for output_idx in range(3):
                ratios.append(three_baseline_losses[output_idx]/three_losses[output_idx])
                
            # backward pass
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # update meters (top1, top5, loss, etc.)
            for name, func in stats_meter.items():
                meters[name].update(func(locals()), input_text.data.shape[0])

            batch_time.update(time.time() - t)

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
                         'avg_data_time'     : data_time.avg,}
                
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
          
        if phase == 'test':
            return results, stats['avg_loss'], stats['avg_loss1'], stats['avg_loss2'], stats['avg_loss3']
        else:
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
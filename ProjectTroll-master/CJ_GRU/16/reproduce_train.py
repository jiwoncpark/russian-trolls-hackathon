
# CJ has its own randState upon calling
# to reproduce results one needs to set
# the internal State of the global stream
# to the one saved when ruuning the code for
# the fist time;
import os,sys,pickle,numpy,random;
CJsavedState = pickle.load(open('CJrandState.pickle','rb'));
numpy.random.set_state(CJsavedState['numpy_CJsavedState']);
random.setstate(CJsavedState['CJsavedState']);
    
sys.path.append('../.');
from utils.device import get_device
from exper.experiment import Experiment
import sys, os, csv
if 'Users' in os.getcwd():
    path = '/Users/zhangyue/Desktop/russian-trolls-hackathon/ProjectTroll-master/data/mydata'
else:
    path = '/home/zyflame104/data'
net_list        = [
                   'GRU',
                   ]
lr_list         = [
                   1e-2, 5e-3, 1e-3, 5e-4, 1e-4
                   ]
data_list       = [1,2,3,4,5]

for net_idx in range(len(net_list)):
    for lr_idx in range(len(lr_list)):
        for data_idx in range(len(data_list)):

            if ( net_idx != 0 or lr_idx != 3 or data_idx != 0 ): continue;
            loader_opts  = {'loader'                    : 'basic_meta_data', # 'binary_classification' for binary classification
                            'data_path'                 : path, #os.path.join(troll_root, 'mydata'),
                            'days'                      : 7,
                            'Glove_name'                : 'twitter.27B',
                            'embedding_dim'             : 25,
                            'fix_length'                : None,
                            'data_num'                  : data_list[data_idx],
                            }
            net_opts     = {'hidden_size'               : 64, #256
                            'attention_size'            : 64,
                            'meta_dim'                  : 9, # not configurable...
                            'meta_hidden_size'          : 16, # something less than 256 but comparable
                            'output_size'               : 3, #1 for binary classification
                            'total_hidden_size'         : 8,
                            'last_sigmoid'              : True, #True for binary classification
                            'output_size'               : 3,
                            }
            train_opts   = {'crit'                      : 'MSELoss', #'MSELoss', #'BCELoss' for binary classification
                            'net'                       : net_list[net_idx],
                            'optim'                     : 'Adam',
                            'weight_decay'              : 5e-4,
                            'optim_kwargs'              : {},
                            'epochs'                    : 100,
                            'lr'                        : lr_list[lr_idx],
                            'milestones_perc'           : [1/3,2/3],
                            'gamma'                     : 0.1,
                            'train_batch_size'          : 31,
                            'test_batch_size'           : 2**5,
                            'device'                    : get_device(),
                            'seed'                      : 0,
                            }
            results_opts = {'training_results_path'     : './results',
                            'train_dump_file'           : 'training_results.json',
                            }
            opts = dict(loader_opts, **net_opts)
            opts = dict(opts, **train_opts)
            opts = dict(opts, **results_opts)
            stats_meter = {'loss': lambda variables: float(variables['loss'].item()),
                           'baseline1': lambda variables: float(variables['baseline1'].item()),
                           'baseline2': lambda variables: float(variables['baseline2'].item()),
                           'baseline3': lambda variables: float(variables['baseline3'].item()),
                            'loss1': lambda variables: float(variables['loss1'].item()),
                           'loss2': lambda variables: float(variables['loss2'].item()),
                           'loss3': lambda variables: float(variables['loss3'].item()),}
            stats_no_meter = {}
            Experiment(opts).run(stats_meter, stats_no_meter)

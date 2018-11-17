#CJ -s '/Users/romano/mydata' '?'

from utils.device import get_device
from exper.experiment import Experiment

net_list        = [
                   'GRU',
                   'LSTM',
                   ]

lr_list         = [
                   1e-2, 5e-3, 1e-3, 5e-4, 1e-4
                    ]

for net_idx in range(len(net_list)):
    for lr_idx in range(len(lr_list)):
        
        loader_opts  = {'loader'                    : 'basic',
                        'data_path'                 : 'data/mydata',
                        'days'                      : 7,
                        'Glove_name'                : 'twitter.27B',
                        'embedding_dim'             : 200,
                        'fix_length'                : None,
                        }
        
        net_opts     = {'hidden_size'               : 256,
                        'output_size'               : 2,
                        }
        
        train_opts   = {'crit'                      : 'MSELoss',
                        'net'                       : net_list[net_idx],
                        'optim'                     : 'Adam',
                        'weight_decay'              : 5e-4,
                        'optim_kwargs'              : {},
                        'epochs'                    : 100,
                        'lr'                        : lr_list[lr_idx],
                        'milestones_perc'           : [1/3,2/3],
                        'gamma'                     : 0.1,
                        'train_batch_size'          : 2**7,
                        'test_batch_size'           : 2**9,
                        'device'                    : get_device(),
                        'seed'                      : 0,
                        }
        
        results_opts = {'training_results_path'     : './results',
                        'train_dump_file'           : 'training_results.json',
                        }
                    
        opts = dict(loader_opts, **net_opts)
        opts = dict(opts, **train_opts)
        opts = dict(opts, **results_opts)
        
        # these meters will be displayed to the console and saved into a csv
        stats_meter = {'loss': lambda variables: float(variables['loss'].item()),
                       }
        
        # these meters will be displayed to the console but not saved into a csv
        stats_no_meter = {}
        
        Experiment(opts).run(stats_meter, stats_no_meter)        

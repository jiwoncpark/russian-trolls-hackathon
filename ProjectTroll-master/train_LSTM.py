#CJ -s '/Users/romano/mydata' '?'

from utils.device import get_device
from exper.experiment import Experiment

net_list        = [
                   'LSTM',
                   ]

lr_list         = [
                   1e-4,
                   ]
troll_root = '/home/jwp/stage/stats285-experiment-management-system/hackathon/ProjectTroll-master'

for net_idx in range(1):
    for lr_idx in range(1):
        
        loader_opts  = {'loader'                    : 'basic',
                        'data_path'                 : troll_root + '/mydata',
                        'days'                      : 7,
                        'Glove_name'                : '6B',
                        'embedding_dim'             : 300,
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
        
        
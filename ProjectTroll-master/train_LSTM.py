#CJ -s '/Users/romano/mydata' '?'

from utils.device import get_device
from exper.experiment import Experiment
import sys, os, csv
# troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
# sys.path.insert(0, troll_root)

# This block of code directs the path of mydata folder to prevent downloading multiple times
# MAKE SURE YOU ALSO UPDATE THE PATH IN LOADER FILE !!!!
if 'Users' in os.getcwd():
    # Specify your local mydata folder
    path = '/Users/zhangyue/Desktop/russian-trolls-hackathon/ProjectTroll-master/data/mydata'
else:
    # Specify your cloud mydata folder
    path = '/scratch/users/yzhang16/Shannon/data/mydata'

# Prepare training log files that show best result.
csv_file = open('best_model_log.csv', mode='w')
fieldnames = ['net_list', 'lr_list', 'wd_list', 'best_test_loss']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

# Specify hyperparatemers
net_list        = [
                   'GRU',
                   ]

lr_list         = [
                   1e-3
                   ]

wd_list         = [
                   #1e-4,
                   5e-4,
                   ]

for net_idx in range(len(net_list)):
    for lr_idx in range(len(lr_list)):
        for wd_idx in range(len(wd_list)):

            loader_opts  = {'loader'                    : 'basic_meta_data', # 'binary_classification' for binary classification
                                                                                   # 'basic_meta_data' for taking in metadata and outputting 3-tuple
                            'data_path'                 : path,
                            'days'                      : 7,
                            'Glove_name'                : 'twitter.27B',
                            'embedding_dim'             : 200,
                            'fix_length'                : None,
                            }

            net_opts     = {'hidden_size'               : 256,
                            'attention_size'            : 256,
                            'meta_dim'                  : 6, # not configurable...
                            'meta_hidden_size'          : 16, # something less than 256 but comparable
                            'output_size'               : 3, #1 for binary classification
                            'total_hidden_size'         : 8,
                            'last_sigmoid'              : True, #True for binary classification
                            'output_size'               : 3,
                            }

            train_opts   = {'crit'                      : 'BCELoss', #'MSELoss', #'BCELoss' for binary classification
                            'net'                       : net_list[net_idx],
                            'optim'                     : 'Adam',
                            'weight_decay'              : wd_list[wd_idx],
                            'optim_kwargs'              : {},
                            'epochs'                    : 30,
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

            best_test_accuracy = Experiment(opts).run(stats_meter, stats_no_meter)
            writer.writerow({'net_list': net_list[net_idx], 'lr_list': lr_list[lr_idx], 
                'wd_list': wd_list[wd_idx], 'best_test_accuracy': best_test_accuracy})

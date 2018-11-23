#CJ -s '/Users/romano/mydata' '?'

from utils.device import get_device
from exper.experiment import Experiment
import sys, os, csv
#troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
#sys.path.insert(0, troll_root)

if 'Users' in os.getcwd():
    # Specify your local mydata folder
    data_path = '/Users/zhangyue/Desktop/russian-trolls-hackathon/ProjectTroll-master/data/mydata'
else:
    # Specify your cloud mydata folder
    data_path = '/home/zyflame104/data'

troll_root = os.path.join(os.environ['REPOROOT'], 'ProjectTroll-master')
sys.path.insert(0, troll_root)
data_path = os.path.join(troll_root, 'mydata')    
    
# Prepare a CSV file for each set of hyperparameters
# csv_file = open('best_model_log.csv', mode='w')
# fieldnames = ['net_list', 'lr_list', 'wd_list', 'best_total_loss', 'best_loss_1', 'best_loss_2', 'best_loss_3']
# writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
# writer.writeheader()

net_list        = [
                   'AttentionLSTM',
                   ]

lr_list         = [
                   1e-2 #, 1e-4
                   ]

wd_list         = [
                   5e-4,
                   ]

for net_idx in range(len(net_list)):
    for lr_idx in range(len(lr_list)):
        for wd_idx in range(len(wd_list)):

            loader_opts  = {'loader'                    : 'basic_meta_data', # 'binary_classification' for binary classification
                                                                                   # 'basic_meta_data' for taking in metadata and outputting 3-tuple
                            'data_path'                 : data_path, #os.path.join(troll_root, 'mydata'),
                            'days'                      : 7,
                            'Glove_name'                : 'twitter.27B',
                            'embedding_dim'             : 25,
                            'fix_length'                : None,
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
                            'weight_decay'              : wd_list[wd_idx],
                            'optim_kwargs'              : {},
                            'epochs'                    : 15,
                            'lr'                        : lr_list[lr_idx],
                            'milestones_perc'           : [1/3,2/3],
                            'gamma'                     : 0.1,
                            'train_batch_size'          : 31,
                            'test_batch_size'           : 31,
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
                           'baseline1': lambda variables: float(variables['baseline1'].item()),
                           'baseline2': lambda variables: float(variables['baseline2'].item()),
                           'baseline3': lambda variables: float(variables['baseline3'].item()),
                           'loss1': lambda variables: float(variables['loss1'].item()),
                           'loss2': lambda variables: float(variables['loss2'].item()),
                           'loss3': lambda variables: float(variables['loss3'].item()),}

            # these meters will be displayed to the console but not saved into a csv
            stats_no_meter = {}

            Experiment(opts).run(stats_meter, stats_no_meter, False)

            #best_result_one_hp = Experiment(opts).run(stats_meter, stats_no_meter)

            # best_total_loss, best_loss_1, best_loss_2, best_loss_3 = best_result_one_hp

            # writer.writerow({'net_list': net_list[net_idx], 'lr_list': lr_list[lr_idx],
            #     'wd_list': wd_list[wd_idx], 'best_total_loss': best_total_loss, 'best_loss_1':
            #     best_loss_1, 'best_loss_2': best_loss_2, 'best_loss_3': best_loss_3})

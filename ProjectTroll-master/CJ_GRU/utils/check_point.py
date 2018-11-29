import torch

def get_check_point_path(obj, epoch):
    return 

def save_check_point(obj, model, loss_type):    
    path = obj.training_results_path+'/'+get_check_point_name(obj)+'-'+loss_type+'.pth'
    torch.save(model.state_dict(), path)    
    print('Checkpoint saved to {}'.format(path))

def get_check_point_name(obj):
    name = 'net=' + obj.net + '-lr=' + str(obj.lr)
    return name


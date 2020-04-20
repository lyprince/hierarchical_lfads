#!/usr/bin/env python

import argparse
import os

import torch
import torchvision
import torchvision.transforms as trf
import torch.optim as opt

from orion.client import report_results

from trainer import RunManager
from scheduler import LFADS_Scheduler
from objective import LFADS_Loss, LogLikelihoodPoisson
from lfads import LFADS_SingleSession_Net
from utils import read_data, load_parameters
from plotter import Plotter


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-p', '--hyperparameter_path', type=str)
parser.add_argument('-o', '--output_dir', default='/tmp', type=str)
parser.add_argument('--max_epochs', default=2000, type=int)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('-t', '--use_tensorboard', action='store_true', default=False)
parser.add_argument('-r', '--restart', action='store_true', default=False)
parser.add_argument('-c', '--do_health_check', action='store_true', default=False)

parser.add_argument('--data_suffix', default='data', type=str)

parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--patience', type=int, default=None)
parser.add_argument('--weight_schedule_dur', type=int, default=None)
parser.add_argument('--kl_max', type=float, default=None)

# parser.add_argument('--config', type=yaml.safe_load)

def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hyperparams = load_parameters(args.hyperparameter_path)
    
    if args.lr:
        hyperparams['optimizer']['lr_init'] = args.lr
        hyperparams['scheduler']['lr_min']  = args.lr * 1e-3
    
    if args.patience:
        hyperparams['scheduler']['scheduler_patience'] = args.patience
        
    if args.weight_schedule_dur:
        hyperparams['objective']['kl']['weight_schedule_dur'] = args.weight_schedule_dur
        hyperparams['objective']['l2']['weight_schedule_dur'] = args.weight_schedule_dur
        
    if args.kl_max:
        hyperparams['objective']['kl']['max'] = args.kl_max
    
    data_name = args.data_path.split('/')[-1]
    model_name = hyperparams['model_name']
    mhp_list = [key.replace('size', '').replace('_', '')[:4] + str(val) for key, val in hyperparams['model'].items() if 'size' in key]
    mhp_list.sort()
    hyperparams['run_name'] = '_'.join(mhp_list) + '_retest'
    save_loc = '%s/%s/%s/%s/'%(args.output_dir, data_name, model_name, hyperparams['run_name'])
    
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    
    data_dict   = read_data(args.data_path)
    train_data  = torch.Tensor(data_dict['train_%s'%args.data_suffix]).to(device)
    valid_data  = torch.Tensor(data_dict['valid_%s'%args.data_suffix]).to(device)
    
    num_trials, num_steps, input_size = train_data.shape
    
    train_ds    = torch.utils.data.TensorDataset(train_data)
    valid_ds    = torch.utils.data.TensorDataset(valid_data)
    train_dl    = torch.utils.data.DataLoader(train_ds, batch_size = args.batch_size, shuffle=True)
    valid_dl    = torch.utils.data.DataLoader(valid_ds, batch_size = valid_data.shape[0])
    
    transforms  = trf.Compose([])
    
    loglikelihood = LogLikelihoodPoisson(dt=float(data_dict['dt']))
    
    objective = LFADS_Loss(loglikelihood            = loglikelihood,
                           loss_weight_dict         = {'kl': hyperparams['objective']['kl'], 
                                                       'l2': hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)

    model = LFADS_SingleSession_Net(input_size           = input_size,
                                    factor_size          = hyperparams['model']['factor_size'],
                                    g_encoder_size       = hyperparams['model']['g_encoder_size'],
                                    c_encoder_size       = hyperparams['model']['c_encoder_size'],
                                    g_latent_size        = hyperparams['model']['g_latent_size'],
                                    u_latent_size        = hyperparams['model']['u_latent_size'],
                                    controller_size      = hyperparams['model']['controller_size'],
                                    generator_size       = hyperparams['model']['generator_size'],
                                    prior                = hyperparams['model']['prior'],
                                    clip_val             = hyperparams['model']['clip_val'],
                                    dropout              = hyperparams['model']['dropout'],
                                    do_normalize_factors = hyperparams['model']['normalize_factors'],
                                    max_norm             = hyperparams['model']['max_norm'],
                                    device               = device).to(device)
    
    total_params = 0
    for ix, (name, param) in enumerate(model.named_parameters()):
        print(ix, name, list(param.shape), param.numel(), param.requires_grad)
        total_params += param.numel()
    
    print('Total parameters: %i'%total_params)

    optimizer = opt.Adam(model.parameters(),
                         lr=hyperparams['optimizer']['lr_init'],
                         betas=hyperparams['optimizer']['betas'],
                         eps=hyperparams['optimizer']['eps'])

    scheduler = LFADS_Scheduler(optimizer      = optimizer,
                                mode           = 'min',
                                factor         = hyperparams['scheduler']['scheduler_factor'],
                                patience       = hyperparams['scheduler']['scheduler_patience'],
                                verbose        = True,
                                threshold      = 1e-4,
                                threshold_mode = 'abs',
                                cooldown       = hyperparams['scheduler']['scheduler_cooldown'],
                                min_lr         = hyperparams['scheduler']['lr_min'])
    
    TIME = torch._np.arange(0, num_steps*data_dict['dt'], data_dict['dt'])
    
    train_truth = {}
    if 'train_rates' in data_dict.keys():
        train_truth['rates'] = data_dict['train_rates']
    if 'train_latent' in data_dict.keys():
        train_truth['latent'] = data_dict['train_latent']
        
    valid_truth = {}
    if 'valid_rates' in data_dict.keys():
        valid_truth['rates'] = data_dict['valid_rates']
    if 'valid_latent' in data_dict.keys():
        valid_truth['latent'] = data_dict['valid_latent']

    plotter = {'train' : Plotter(time=TIME, truth=train_truth),
               'valid' : Plotter(time=TIME, truth=valid_truth)}
    
    if args.use_tensorboard:
        import importlib
        if importlib.util.find_spec('torch.utils.tensorboard'):
            tb_folder = save_loc + 'tensorboard/'
            if not os.path.exists(tb_folder):
                os.mkdir(tb_folder)
            elif os.path.exists(tb_folder) and args.restart:
                os.system('rm -rf %s'%tb_folder)
                os.mkdir(tb_folder)

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tb_folder)
            rm_plotter = plotter
        else:
            writer = None
            rm_plotter = None
    else:
        writer = None
        rm_plotter = None

    run_manager = RunManager(model      = model,
                             objective  = objective,
                             optimizer  = optimizer,
                             scheduler  = scheduler,
                             train_dl   = train_dl,
                             valid_dl   = valid_dl,
                             transforms = transforms,
                             writer     = writer,
                             plotter    = rm_plotter,
                             max_epochs = args.max_epochs,
                             save_loc   = save_loc,
                             do_health_check = args.do_health_check)

    run_manager.run()
    
    report_results([dict(name= 'valid_loss',
                         type= 'objective',
                         value= run_manager.best)])

    fig_folder = save_loc + 'figs/'
    
    if os.path.exists(fig_folder):
        os.system('rm -rf %s'%fig_folder)
    os.mkdir(fig_folder)
    
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')
    fig_dict = plotter['valid'].plot_summary(model = run_manager.model, dl=run_manager.valid_dl)
    for k, v in fig_dict.items():
        if type(v) == Figure:
            v.savefig(fig_folder+k+'.svg')
    
if __name__ == '__main__':
    main()
#!/usr/bin/env python

import argparse
import os

import torch
import torchvision
import torch.optim as opt
import torchvision.transforms as trf

from orion.client import report_results

from synthetic_data import SyntheticCalciumVideoDataset

from trainer import RunManager
from scheduler import LFADS_Scheduler
from objective import Conv_LFADS_Loss, LogLikelihoodGaussian
from conv_lfads import Conv3d_LFADS_Net
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

parser.add_argument('--lr', type=float, default=None)

def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hyperparams = load_parameters(args.hyperparameter_path)
    
    orion_hp_string = ''
    if args.lr:
        lr = args.lr
        hyperparams['optimizer']['lr_init'] = lr
        hyperparams['scheduler']['lr_min']  = lr * 1e-3
        orion_hp_string += 'lr= %.4f\n'%lr
    
    data_name = args.data_path.split('/')[-1]
    model_name = args.hyperparameter_path.split['/'][-1]
    mhp_list = [key.replace('size', '').replace('deep', 'd').replace('obs', 'o').replace('_', '')[:4] + str(val) for key, val in hyperparams['model'].items() if 'size' in key]
    mhp_list.sort()
    hyperparams['run_name'] = '_'.join(mhp_list)
    orion_hp_string = orion_hp_string.replace('\n', '-').replace(' ', '').replace('=', '')
    orion_hp_string = '_orion-'+orion_hp_string
    hyperparams['run_name'] += orion_hp_string
    save_loc = '%s/%s/%s/%s/'%(args.output_dir, data_name, model_name, hyperparams['run_name'])

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        
    # Setup DataLoader goes here
    data_dict   = read_data(args.data_path)
    train_dl    = torch.utils.data.DataLoader(SyntheticCalciumVideoDataset(traces= data_dict['train_fluor'], cells=data_dict['cells'], device=device), batch_size=args.batch_size)
    valid_dl    = torch.utils.data.DataLoader(SyntheticCalciumVideoDataset(traces= data_dict['valid_fluor'], cells=data_dict['cells'], device=device), batch_size=args.batch_size)
    
    num_trials, num_steps, num_cells = data_dict['train_fluor'].shape
    num_cells, width, height = data_dict['cells'].shape
    
    model = Conv3d_LFADS_Net(input_dims      = (num_steps, width, height),
                             channel_dims    = hyperparams['model']['channel_dims'],
                             factor_size     = hyperparams['model']['factor_size'],
                             g_encoder_size  = hyperparams['model']['g_encoder_size'],
                             c_encoder_size  = hyperparams['model']['c_encoder_size'],
                             g_latent_size   = hyperparams['model']['g_latent_size'],
                             u_latent_size   = hyperparams['model']['u_latent_size'],
                             controller_size = hyperparams['model']['controller_size'],
                             generator_size  = hyperparams['model']['generator_size'],
                             prior           = hyperparams['model']['prior'],
                             clip_val        = hyperparams['model']['clip_val'],
                             conv_dropout    = hyperparams['model']['conv_dropout'],
                             lfads_dropout   = hyperparams['model']['lfads_dropout'],
                             do_normalize_factors = hyperparams['model']['normalize_factors'],
                             max_norm        = hyperparams['model']['max_norm'],
                             device          = device).to(device)
    
    model.to(dtype=train_dl.dataset.dtype)
    torch.set_default_dtype(train_dl.dataset.dtype)
    
    transforms = trf.Compose([])
    
    loglikelihood = LogLikelihoodGaussian()
    objective = Conv_LFADS_Loss(loglikelihood=loglikelihood,
                                loss_weight_dict={'kl': hyperparams['objective']['kl'],
                                                  'l2': hyperparams['objective']['l2']},
                                                   l2_con_scale= hyperparams['objective']['l2_con_scale'],
                                                   l2_gen_scale= hyperparams['objective']['l2_gen_scale']).to(device)
    
    total_params = 0
    for ix, (name, param) in enumerate(model.named_parameters()):
        print(ix, name, list(param.shape), param.numel(), param.requires_grad)
        total_params += param.numel()
    
    print('Total parameters: %i'%total_params)
    
    optimizer = opt.Adam([p for p in model.parameters() if p.requires_grad],
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
    if 'train_latent' in data_dict.keys():
        train_truth['latent'] = data_dict['train_latent']
        
    valid_truth = {}
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
    
    import matplotlib
    matplotlib.use('Agg')
    fig_dict = plotter['valid'].plot_summary(model = run_manager.model, dl=run_manager.valid_dl, mode='video', num_average=4)
    for k, v in fig_dict.items():
        if type(k) == matplotlib.figure.Figure:
            v.savefig(fig_folder+k+'.svg')
    
if __name__ == '__main__':
    main()
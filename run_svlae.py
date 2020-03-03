#!/usr/bin/env python

import argparse
import os

import torch
import torchvision
import torch.optim as opt
import torchvision.transforms as trf

from orion.client import report_results

from trainer import RunManager
from scheduler import LFADS_Scheduler
from objective import SVLAE_Loss, LogLikelihoodPoisson, LogLikelihoodPoissonSimplePlusL1, LogLikelihoodGaussian
from svlae import SVLAE_Net
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
parser.add_argument('--log10_lr', type=float, default=None)
parser.add_argument('--kl_deep_max', type=float, default=None)
parser.add_argument('--kl_obs_max', type=float, default=None)
parser.add_argument('--kl_obs_dur', type=int, default=None)
parser.add_argument('--kl_obs_dur_scale', type=int, default=1.0)
parser.add_argument('--deep_start_p', type=int, default=None)
parser.add_argument('--deep_start_p_scale', type=float, default=1.0)
parser.add_argument('--l2_gen_scale', type=float, default=None)
parser.add_argument('--l2_con_scale', type=float, default=None)
parser.add_argument('--log10_l2_gen_scale', type=float, default=None)
parser.add_argument('--log10_l2_con_scale', type=float, default=None)


def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hyperparams = load_parameters(args.hyperparameter_path)
    
    orion_hp_string = ''
    if args.lr or args.log10_lr:
        if args.log10_lr:
            lr = 10**args.log10_lr
        else:
            lr = args.lr
        hyperparams['optimizer']['lr_init'] = lr
        hyperparams['scheduler']['lr_min']  = lr * 1e-3
        orion_hp_string += 'lr= %.4f\n'%lr
        
    if args.kl_obs_dur:
        hyperparams['objective']['kl_obs']['schedule_dur'] = args.kl_obs_dur * args.kl_obs_dur_scale
        orion_hp_string += 'kl_obs_dur= %i\n'%(args.kl_obs_dur*args.kl_obs_dur_scale)

    if args.kl_obs_max:
        hyperparams['objective']['kl_obs']['max'] = args.kl_obs_max
        orion_hp_string += 'kl_obs_max= %.3f\n'%(args.kl_obs_max)
        
    if args.kl_deep_max:
        hyperparams['objective']['kl_deep']['max'] = args.kl_deep_max
        orion_hp_string += 'kl_deep_max= %.3f\n'%(args.kl_deep_max)
    
    if args.deep_start_p:
        print('deep_start found')
        deep_start = int(args.deep_start_p * args.deep_start_p_scale * hyperparams['objective']['kl_obs']['schedule_dur'])
        hyperparams['objective']['kl_deep']['schedule_start'] = deep_start
        hyperparams['objective']['l2']['schedule_start'] = deep_start
        hyperparams['model']['unfreeze'] = deep_start
        orion_hp_string += 'deep_start= %i\n'%deep_start
        
    if args.l2_gen_scale or args.log10_l2_gen_scale:
        if args.log10_l2_gen_scale:
            l2_gen_scale = 10**args.log10_l2_gen_scale
        else:
            l2_gen_scale = args.l2_gen_scale
        hyperparams['objective']['l2_gen_scale'] = l2_gen_scale
        orion_hp_string += 'l2_gen_scale= %.3f\n'%l2_gen_scale
    
    if args.l2_con_scale or args.log10_l2_con_scale:
        if args.log10_l2_con_scale:
            l2_con_scale = 10**args.log10_l2_con_scale
        else:
            l2_con_scale = args.l2_con_scale
        hyperparams['objective']['l2_con_scale'] = l2_con_scale
        orion_hp_string += 'l2_con_scale= %.3f\n'%l2_con_scale
    
    data_name = args.data_path.split('/')[-1]
    model_name = hyperparams['model_name']
    mhp_list = [key.replace('size', '').replace('deep', 'd').replace('obs', 'o').replace('_', '')[:4] + str(val) for key, val in hyperparams['model'].items() if 'size' in key]
    mhp_list.sort()
    hyperparams['run_name'] = '_'.join(mhp_list)
    orion_hp_string = orion_hp_string.replace('\n', '-').replace(' ', '')
    orion_hp_string = '_orion('+orion_hp_string+')'
    hyperparams['run_name'] += orion_hp_string
    save_loc = '%s/%s/%s/%s/'%(args.output_dir, data_name, model_name, hyperparams['run_name'])

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    
    data_dict   = read_data(args.data_path)
    train_data  = torch.Tensor(data_dict['train_fluor']).to(device)
    valid_data  = torch.Tensor(data_dict['valid_fluor']).to(device)
    
    num_trials, num_steps, input_size = train_data.shape
    
    train_ds    = torch.utils.data.TensorDataset(train_data)
    valid_ds    = torch.utils.data.TensorDataset(valid_data)
    train_dl    = torch.utils.data.DataLoader(train_ds, batch_size = args.batch_size, shuffle=True)
    valid_dl    = torch.utils.data.DataLoader(valid_ds, batch_size = valid_data.shape[0])
    
#     transforms = trf.Compose([trf.Normalize(mean=(train_data.mean(),), std=(train_data.std(),))])
    transforms = trf.Compose([])
    
    loglikelihood_obs  = LogLikelihoodGaussian()
    loglikelihood_deep = LogLikelihoodPoissonSimplePlusL1(dt=float(data_dict['dt']))
    
    objective = SVLAE_Loss(loglikelihood_obs        = loglikelihood_obs,
                           loglikelihood_deep       = loglikelihood_deep,
                           loss_weight_dict         = {'kl_deep': hyperparams['objective']['kl_deep'],
                                                       'kl_obs' : hyperparams['objective']['kl_obs'],
                                                       'l2'     : hyperparams['objective']['l2']},
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)
    
    hyperparams['model']['obs']['tau']['value']/=data_dict['dt']
    
    model = SVLAE_Net(input_size            = input_size,
                      factor_size           = hyperparams['model']['factor_size'],
                      obs_encoder_size      = hyperparams['model']['obs_encoder_size'],
                      obs_latent_size       = hyperparams['model']['obs_latent_size'],
                      obs_controller_size   = hyperparams['model']['obs_controller_size'],
                      deep_g_encoder_size   = hyperparams['model']['deep_g_encoder_size'],
                      deep_c_encoder_size   = hyperparams['model']['deep_c_encoder_size'],
                      deep_g_latent_size    = hyperparams['model']['deep_g_latent_size'],
                      deep_u_latent_size    = hyperparams['model']['deep_u_latent_size'],
                      deep_controller_size  = hyperparams['model']['deep_controller_size'],
                      generator_size        = hyperparams['model']['generator_size'],
                      prior                 = hyperparams['model']['prior'],
                      clip_val              = hyperparams['model']['clip_val'],
                      generator_burn        = hyperparams['model']['generator_burn'],
                      dropout               = hyperparams['model']['dropout'],
                      do_normalize_factors  = hyperparams['model']['normalize_factors'],
                      factor_bias           = hyperparams['model']['factor_bias'],
                      max_norm              = hyperparams['model']['max_norm'],
                      deep_freeze           = hyperparams['model']['deep_freeze'],
                      unfreeze              = hyperparams['model']['unfreeze'],
                      obs_params            = hyperparams['model']['obs'],
                      device                = device).to(device)
    
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

    plotter = {'train' : Plotter(time=TIME, truth={'rates'   : data_dict['train_rates'],
                                                   'spikes'  : data_dict['train_spikes'],
                                                   'latent'  : data_dict['train_latent']}),
               'valid' : Plotter(time=TIME, truth={'rates'   : data_dict['valid_rates'],
                                                   'spikes'  : data_dict['valid_spikes'],
                                                   'latent'  : data_dict['valid_latent']})}
    
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
    
    if not torch._np.isfinite(run_manager.best):
        run_manager.best = 1e8
    
    report_results([dict(name= 'valid_loss',
                         type= 'objective',
                         value= run_manager.best)])

if __name__ == '__main__':
    main()

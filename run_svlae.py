import argparse
import os

import torch
import torchvision
import torch.optim as opt
import torchvision.transforms as trf

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

def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hyperparams = load_parameters(args.hyperparameter_path)
    
    data_name = args.data_path.split('/')[-1]
    model_name = hyperparams['model_name']
    mhp_list = [key.replace('size', '').replace('_', '')[:4] + str(val) for key, val in hyperparams['model'].items() if 'size' in key]
    mhp_list.sort()
    hyperparams['run_name'] = '_'.join(mhp_list)
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
                           kl_obs_weight_init           = hyperparams['objective']['kl_obs_weight_init'],
                           kl_obs_weight_schedule_dur   = hyperparams['objective']['kl_obs_weight_schedule_dur'],
                           kl_obs_weight_schedule_start = hyperparams['objective']['kl_obs_weight_schedule_start'],
                           kl_obs_weight_max            = hyperparams['objective']['kl_obs_weight_max'],
                           kl_deep_weight_init           = hyperparams['objective']['kl_deep_weight_init'],
                           kl_deep_weight_schedule_dur   = hyperparams['objective']['kl_deep_weight_schedule_dur'],
                           kl_deep_weight_schedule_start = hyperparams['objective']['kl_deep_weight_schedule_start'],
                           kl_deep_weight_max            = hyperparams['objective']['kl_deep_weight_max'],
                           l2_weight_init           = hyperparams['objective']['l2_weight_init'],
                           l2_weight_schedule_dur   = hyperparams['objective']['l2_weight_schedule_dur'],
                           l2_weight_schedule_start = hyperparams['objective']['l2_weight_schedule_start'],
                           l2_weight_max            = hyperparams['objective']['l2_weight_max'],
                           l2_con_scale             = hyperparams['objective']['l2_con_scale'],
                           l2_gen_scale             = hyperparams['objective']['l2_gen_scale']).to(device)
    
    hyperparams['model']['obs']['tau']['value']/=data_dict['dt']
    
    model = SVLAE_Net(input_size            = input_size,
                      factor_size           = hyperparams['model']['factor_size'],
                      obs_encoder_size      = hyperparams['model']['obs_encoder_size'],
                      obs_latent_size       = hyperparams['model']['obs_latent_size'],
                      obs_controller_size   = hyperparams['model']['obs_controller_size'],
                      deep_encoder_size     = hyperparams['model']['deep_encoder_size'],
                      deep_encoder_size     = hyperparams['model']['deep_encoder_size'],
                      deep_latent_size      = hyperparams['model']['deep_latent_size'],
                      deep_latent_size      = hyperparams['model']['deep_latent_size'],
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

if __name__ == '__main__':
    main()
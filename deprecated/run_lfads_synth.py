from models import *
from utils import read_data, load_parameters, save_parameters
import argparse
import time
import yaml
import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-p', '--parameter_path', type=str)
# parser.add_argument('-c', '--checkpoint', default=None, type=str)
parser.add_argument('--max_epochs', default=2000, type=int)
parser.add_argument('--batch_size', default=None, type=int)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device, flush=True)
    args = parser.parse_args()
    data_name = args.data_path.split('/')[-1]
    _, system_name, model_name = args.parameter_path.split('/')[-1].split('.')[0].split('_')

    # Load hyperparameters
    hyperparams = load_parameters(args.parameter_path)

    # Alter run name to describe seed, parameter settings and date of model run
    hyperparams['run_name'] += '_%s'%data_name
    hyperparams['run_name'] += '_f%i_g1%i_eg1%i_u%i'%(hyperparams['factors_dim'], hyperparams['g_dim'], hyperparams['g0_encoder_dim'], hyperparams['u_dim'])

    if hyperparams['u_dim'] > 0:
        hyperparams['run_name'] += '_c1%i_ec1%i'%(hyperparams['c_controller_dim'], hyperparams['c_encoder_dim'])

    if model_name == 'ladder':
        hyperparams['run_name'] += '_g2%i_c2%i_eg2%i_ec2%i'%(hyperparams['h_dim'], hyperparams['a_controller_dim'],
                                                             hyperparams['h0_encoder_dim'], hyperparams['a_encoder_dim'])
    elif model_name in ['gaussian', 'edgeworth']:
        hyperparams['run_name'] += '_k%i'%hyperparams['kernel_dim']

    hyperparams['run_name'] += '_%s'%time.strftime('%y%m%d')
    save_parameters(hyperparams, output=args.output)

    # Load data
    data_dict = read_data(args.data_path)
    datatype = model_name if model_name in ['spikes', 'oasis'] else 'fluor'

    train_data = torch.Tensor(data_dict['train_spikes']).to(device)
    valid_data = torch.Tensor(data_dict['valid_spikes']).to(device)

    train_truth = {'rates'  : data_dict['train_rates']}

    valid_truth = {'rates'  : data_dict['valid_rates']}

    if model_name == 'ladder':
        train_truth['spikes'] = data_dict['train_spikes']
        valid_truth['spikes'] = data_dict['valid_spikes']

    if 'train_latent' in data_dict.keys():
        train_truth['latent'] = data_dict['train_latent']

    if 'valid_latent' in data_dict.keys():
        valid_truth['latent'] = data_dict['valid_latent']

    train_ds      = torch.utils.data.TensorDataset(train_data)
    valid_ds      = torch.utils.data.TensorDataset(valid_data)

    num_trials, num_steps, num_cells = train_data.shape;
    print('Data dimensions: N=%i, T=%i, C=%i'%train_data.shape, flush=True);
    print('Number of datapoints = %s'%train_data.numel(), flush=True)

    # Initialize Network
    Net = LFADS if model_name in ['spikes', 'oasis'] else MomentLFADS if model_name in ['gaussian', 'edgeworth'] else LadderLFADS
    model = Net(inputs_dim = num_cells, T = num_steps, dt = float(data_dict['dt']), device=device,
                 model_hyperparams=hyperparams).to(device)

    # Train Network
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = int(num_trials/25)

    total_params = 0
    for ix, (name, param) in enumerate(model.named_parameters()):
        print(ix, name, list(param.shape), param.numel(), param.requires_grad)
        total_params += param.numel()

    print('Total parameters: %i'%total_params)

    model.fit(train_dataset=train_ds, valid_dataset=valid_ds,
          train_truth=train_truth, valid_truth=valid_truth,
          max_epochs=args.max_epochs, batch_size=batch_size,
          use_tensorboard=False, health_check=False, home_dir=args.output)

if __name__ == '__main__':
    main()

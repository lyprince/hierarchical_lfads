#!/usr/bin/env python

import argparse
import os
import pickle
import yaml

import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')

from utils import write_data, read_data, load_parameters, batchify_sample
from train_model import prep_model
from conv_lfads import Conv3d_LFADS_Net
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model_dir', type=str)
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-n', '--num_average', default=200, type=int)
parser.add_argument('--data_suffix', default='data', type=str)

def main():
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_name = args.model_dir.split('/')[-3].split('_')[0]
    data_name = args.model_dir.split('/')[-4]
    
    hp_path = args.model_dir + 'hyperparameters.yaml'
    hyperparams = load_parameters(hp_path)
    
    data_dict = read_data(args.data_path)
    
    # Re-instantiate model
    train_dl, valid_dl, plotter, model, objective = prep_model(model_name  = model_name,
                                                               data_dict   = data_dict,
                                                               data_suffix = args.data_suffix,
                                                               batch_size  = args.num_average,
                                                               device = device,

    # Load parameters

    state_dict = torch.load(args.model_dir + 'checkpoints/best.pth')
    
    model.load_state_dict(state_dict['net'])
    model.eval()
    
    # Dictionary for storing inferred states
    latent_dict = {'train' : {}, 'valid' : {}}
    latent_dict['train']['latent'] = []
    latent_dict['valid']['latent'] = []
    latent_dict['train']['recon'] = []
    latent_dict['valid']['recon'] = []
    if model_name == 'svlae' or model_name =='svlae-nopoisson':
        latent_dict['train']['spikes'] = []
        latent_dict['valid']['spikes'] = []

    with torch.no_grad():
        
        for dl, key in ((train_dl, 'train'), (valid_dl, 'valid')):

            latent_dict[key]['latent'] = []
            latent_dict[key]['rates'] = []
            if model_name == 'svlae' or model_name =='svlae-nopoisson':

                latent_dict[key]['spikes'] = []
                latent_dict[key]['fluor'] = []
            for x in dl.dataset:
                x = x[0]
                result = infer_and_recon(x, batch_size=args.num_average, model=model)
                latent_dict[key]['latent'].append(result['latent'])
                latent_dict[key]['rates'].append(result['rates'])
                if model_name == 'svlae' or model_name == 'svlae-nopoisson':

                    latent_dict[key]['spikes'].append(result['spikes'])
                    latent_dict[key]['fluor'].append(result['fluor'])
                    
                if 'inputs' in result.keys():
                    if 'inputs' not in latent_dict[key].keys():
                        latent_dict[key]['inputs'] = []
                    latent_dict[key]['inputs'].append(result['inputs'])
                    
            if args.data_suffix == 'ospikes':
                latent_dict[key]['spikes'] = data_dict[key+'_ospikes']
                latent_dict[key]['fluor'] = data_dict[key+'_ocalcium']
    
    for dataset, latent_dict_k in latent_dict.items():
        for variable, val in latent_dict_k.items():
            latent_dict[dataset][variable] = np.array(latent_dict[dataset][variable])
    
    truth_dict = {}
    for key in ['train', 'valid']:
        truth_dict[key] = {}
        for var in ['rates', 'spikes', 'latent', 'fluor']:
            data_dict_key = key + '_' + var
            
            if data_dict_key in data_dict.keys():
                truth_dict[key][var] = data_dict[data_dict_key]
                if var == 'latent':
                    if data_dict_key == 'train_latent':
                        L_train = fit_linear_model(np.concatenate(latent_dict[key][var][:, 10:]),
                                                   np.concatenate(truth_dict[key][var][:, 10:]))
                    latent_dict[key]['latent_aligned'] = L_train.predict(np.concatenate(latent_dict[key][var]))
                    truth_dict[key]['latent_aligned'] = np.concatenate(truth_dict[key]['latent'])
    
    results_dict = {}
    figs_dict = {}
    for key in ['train', 'valid']:
        results_dict[key], figs_dict[key] = compare_truth(latent_dict[key], truth_dict[key])
#         print(results_dict.keys())
#         pdb.set_trace()
        for var, sub_dict in figs_dict[key].items():

            sub_dict['fig'].savefig(args.model_dir + 'figs/%s_%s_rsq.svg'%(key, var))
#             print(type(sub_dict['fig']))
            print('saved figure at ' + args.model_dir + 'figs/%s_%s_rsq.svg'%(key, var))
                
    factor_size = model.factor_size
    if hasattr(model, 'u_latent_size'):
        u_size = model.u_latent_size
    elif hasattr(model, 'lfads'):
        u_size = model.lfads.u_latent_size
    elif hasattr(model, 'deep_model'):
        u_size = model.deep_model.u_latent_size

    train_size, steps_size, state_size = data_dict['train_%s'%args.data_suffix].shape
    valid_size, steps_size, state_size = data_dict['valid_%s'%args.data_suffix].shape

    data_size = train_size + valid_size

    factors = np.zeros((data_size, steps_size, factor_size))
    rates   = np.zeros((data_size, steps_size, state_size))
        
    if 'train_idx' in data_dict.keys() and 'valid_idx' in data_dict.keys():
        
        if u_size > 0:
            inputs  = np.zeros((data_size, steps_size, state_size))

        if model_name == 'svlae' or model_name=='svlae-nopoisson' or args.data_suffix == 'ospikes':
            spikes  = np.zeros((data_size, steps_size, state_size))
            fluor   = np.zeros((data_size, steps_size, state_size))
        
        train_idx = data_dict['train_idx']
        valid_idx = data_dict['valid_idx']
        
        latent_dict['ordered'] = {}

        factors[train_idx] = latent_dict['train']['latent']
        factors[valid_idx] = latent_dict['valid']['latent']
        latent_dict['ordered']['factors'] = factors

        rates[train_idx] = latent_dict['train']['rates']
        rates[valid_idx] = latent_dict['valid']['rates']
        latent_dict['ordered']['rates'] = rates

        if u_size > 0:
            inputs[train_idx] = latent_dict['train']['inputs']
            inputs[valid_idx] = latent_dict['valid']['inputs']
            latent_dict['ordered']['inputs'] = inputs

        if model_name == 'svlae' or model_name =='svlae-nopoisson' or args.data_suffix == 'ospikes':
            spikes[train_idx] = latent_dict['train']['spikes']
            spikes[valid_idx] = latent_dict['valid']['spikes']
            latent_dict['ordered']['spikes'] = spikes
            fluor[train_idx] = latent_dict['train']['fluor']
            fluor[valid_idx] = latent_dict['valid']['fluor']
            latent_dict['ordered']['fluor'] = fluor
        
    if factor_size == 3:
        for key in ['train', 'valid']:
            fig = plot_3d(X=latent_dict[key]['latent_aligned'].T, title='rsq= %.3f'%results_dict[key]['latent_aligned']['rsq']) #latent_dict[key]['latent_aligned'].T
            fig.savefig(args.model_dir + 'figs/%s_factors3d_rsq.svg'%(key))
        
    pickle.dump(latent_dict, file=open('%slatent.pkl'%args.model_dir, 'wb'))
    yaml.dump(results_dict, open('%sresults.yaml'%args.model_dir, 'w'), default_flow_style=False)
    
def infer_and_recon(sample, batch_size, model):

    result = {}
    batch = batchify_sample(sample, batch_size)
    if isinstance(model,Conv3d_LFADS_Net):
        recon, (factors, inputs), _, _, cout = model(batch)
        result['convout'] = cout
    else:
        recon, (factors, inputs) = model(batch)
        
    
    result['latent'] = factors.mean(dim=1).cpu().numpy()
    if 'rates' in recon.keys():
        result['rates'] = recon['rates'].mean(dim=1).cpu().numpy()
    if inputs is not None:
        result['inputs'] = inputs.mean(dim=1).cpu().numpy()
    if 'spikes' in recon.keys():
        result['spikes'] = recon['spikes'].mean(dim=1).cpu().numpy()
        if isinstance(model,Conv3d_LFADS_Net):
            result['fluor'] = result['convout'].mean(dim=0).cpu().numpy()
        else:
            result['fluor'] = recon['data'].mean(dim=0).cpu().numpy()
    return result
    
from sklearn.linear_model import LinearRegression

def align_linear(x, y=None, L=None):
    if L is not None:
        return L.predict(x)
    elif y is not None:
        return fit_linear_model(x, y).predict(x)
    L = fit_linear_model(x, y)

def fit_linear_model(x, y):
    L = LinearRegression().fit(x, y)
    return L
    
def compute_rsquared(x, y, model=None):
    
    if model is not None:
        return model(x, y).score()
    else:
        return np.corrcoef(x, y)[0, 1]**2
        

def plot_3d(X, Y=None, figsize = (12, 12), view = (None, None), title=None):

    '''TBC'''

    assert X.shape[0] == 3, 'X data must be 3 dimensional'

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0], X[1], X[2], lw=0.1)
    if Y:
        assert Y.shape[0] == 3, 'Y data must be 3 dimensional'
        ax.plot(Y[0], Y[1], Y[2], lw=0.1, alpha=0.7)
    ax.view_init(view[0], view[1])
    if title:
        ax.set_title(title, fontsize=16)
    
    return fig
    
def plot_rsquared(x, y, figsize=(4,4), ms=1, title=''):

    '''
    TBC
    '''

    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, '.', ms=ms, color='dimgrey', rasterized=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Reconstruction')
    plt.ylabel('Truth')

    return fig

def compare_truth(latent_dict, truth_dict, model_name):
    results_dict = {}
    figs_dict = {}
    def compare(key, x_dict, y_dict, save=True):
        results_dict = {}
        figs_dict = {}
        results_dict['rsq'] = compute_rsquared(x= x_dict[key].flatten(),
                                               y= y_dict[key].flatten())
        figs_dict['fig'] = plot_rsquared(x_dict[key].flatten(),
                                            y_dict[key].flatten(),
                                            title='rsq= %.3f'%results_dict['rsq'])
        
        return results_dict, figs_dict
    
    for var in ['rates', 'spikes', 'fluor', 'latent_aligned']:

#         print(latent_dict.keys())
#         print(truth_dict.keys())
        if var in latent_dict.keys() and var in truth_dict.keys():
            results_dict[var], figs_dict[var] = compare(var, latent_dict, truth_dict)
            
    return results_dict, figs_dict

    
if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb

from utils import batchify_random_sample

class Plotter(object):
    def __init__(self, time, truth=None, base_fontsize=14):
        self.dt = np.diff(time)[0]
        self.time = time
        self.fontsize={'ticklabel' : base_fontsize-2,
                       'label'     : base_fontsize,
                       'title'     : base_fontsize+2,
                       'suptitle'  : base_fontsize+4}
        
        self.colors = {'linc_red'  : '#E84924',
                       'linc_blue' : '#37A1D0'}
        
        self.truth = truth
    #------------------------------------------------------------------------------
    
    def plot_summary(self, model, data, num_average=100, ix=None):
        
        '''
        plot_summary(data, truth=None, num_average=100, ix=None)
        
        Plot summary figures for dataset and ground truth if available. Create a batch
        from one sample by repeating a certain number of times, and average across them.
        
        Arguments:
            - data (torch.Tensor) : dataset
            - truth (dict) : ground truth dictionary
            - num_average (int) : number of samples from posterior to average over
            - ix (int) : index of data samples to make summary plot from
            
        Returns:
            - fig_dict : dict of summary figures
        '''
        
        plt.close()
        
        figs_dict = {}
        
        batch_example, ix = batchify_random_sample(data=data, batch_size=num_average, ix=ix)
        figs_dict['ix'] = ix        
        
        model.eval()
        with torch.no_grad():
            recon, factors = model(batch_example)
        orig = data[ix].cpu().numpy()
        
#         pdb.set_trace()
        figs_dict['traces'] = self.plot_traces(recon['data'].mean(dim=0).detach().cpu().numpy(), orig, mode='activity', norm=False)
        figs_dict['traces'].suptitle('Actual fluorescence trace vs.\nestimated mean for a sampled trial')
        
        if self.truth:
            if 'rates' in self.truth.keys():
                recon_rates = recon['rates'].mean(dim=0).cpu().numpy()
                true_rates  = self.truth['rates'][ix]
                figs_dict['truth_rates'] = self.plot_traces(recon_rates, true_rates, mode='rand')
                figs_dict['truth_rates'].suptitle('Reconstructed vs ground-truth rate function')
            
            if 'latent' in self.truth.keys():
                pred_factors = factors.mean(dim=0).cpu().numpy()
                true_factors = self.truth['latent'][ix]
                figs_dict['truth_factors'] = self.plot_traces(pred_factors, true_factors, num_traces=true_factors.shape[-1], ncols=1)
                figs_dict['truth_factors'].suptitle('Reconstructed vs ground-truth factors')
            else:
                figs_dict['factors'] = self.plot_factors(factors.cpu().numpy())
        
        else:
            figs_dict['factors'] = self.plot_factors(factors.cpu().numpy())
        
        if hasattr(model, 'u_posterior_mean'):
            figs_dict['inputs'] = self.plot_inputs(model.u_posterior_mean.cpu.numpy())
        return figs_dict
    
    #------------------------------------------------------------------------------W
    #------------------------------------------------------------------------------
    
    def plot_traces(self, pred, true, figsize=(8,8), num_traces=12, ncols=2, mode=None, norm=True, pred_logvar=None):
        '''
        Plot trace and compare to ground truth
        
        Arguments:
            - pred (np.array): array of predicted values to plot (dims: num_steps x num_cells)
            - true (np.array)   : array of true values to plot (dims: num_steps x num_cells)
            - figsize (2-tuple) : figure size (width, height) in inches (default = (8, 8))
            - num_traces (int)  : number of traces to plot (default = 24)
            - ncols (int)       : number of columns in figure (default = 2)
            - mode (string)     : mode to select subset of traces. Options: 'activity', 'rand', None.
                                  'Activity' plots the the num_traces/2 most active traces and num_traces/2
                                  least active traces defined sorted by mean value in trace
            - norm (bool)       : normalize predicted and actual values (default=True)
            - pred_logvar (np.array) : array of predicted values log-variance (dims: num_steps x num_cells) (default= None)
        
        '''
        
        num_cells = pred.shape[-1]
        
        nrows = int(num_traces/ncols)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        axs = np.ravel(axs)
        
        if mode == 'rand':  
            idxs  = np.random.choice(list(range(num_cells)), size=num_traces, replace=False)
            idxs.sort()
                
        elif mode == 'activity':
            idxs = true.max(axis=0).argsort()[-num_traces:]
        
        else:
            idxs  = list(range(num_cells))
        
        for ii, (ax,idx) in enumerate(zip(axs,idxs)):
            plt.sca(ax)
            plt.plot(self.time, true[:, idx], lw=2, color=self.colors['linc_red'])
            plt.plot(self.time, pred[:, idx], lw=2, color=self.colors['linc_blue'])
                
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.legend(['Actual', 'Reconstructed'])
        
        return fig
    
    #------------------------------------------------------------------------------
    
    def plot_factors(self, factors, max_in_col=5, figsize=(8,8)):
        
        '''
        plot_factors(max_in_col=5, figsize=(8,8))
        
        Plot inferred factors in a grid
        
        Arguments:
            - max_in_col (int) : maximum number of subplots in a column
            - figsize (tuple of 2 ints) : figure size in inches
        Returns
            - figure
        '''
        
        batch_size, steps_size, factors_size = factors.shape
        
        nrows = min(max_in_col, factors_size)
        ncols = int(np.ceil(factors_size/max_in_col))
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        axs = np.ravel(axs)
        factors = factors.mean(dim=0).cpu().numpy()
        fmin = factors.min()
        fmax = factors.max()
        
        for jx in range(factors_size):
            plt.sca(axs[jx])
            plt.plot(self.time, factors[:, jx])
            plt.ylim(fmin-0.1, fmax+0.1)
            
            if jx%ncols == 0:
                plt.ylabel('Activity')
            else:
                plt.ylabel('')
                axs[jx].set_yticklabels([])
            
            if (jx - jx%ncols)/ncols == (nrows-1):
                plt.xlabel('Time (s)')
            else:
                plt.xlabel('')
                axs[jx].set_xticklabels([])
        
        fig.suptitle('Factors 1-%i for a sampled trial.'%factors.shape[1])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        
        return fig
    
    #------------------------------------------------------------------------------
    
    def plot_inputs(self, inputs, fig_width=8, fig_height=1.5):
        
        '''
        plot_inputs(fig_width=8, fig_height=1.5)
        
        Plot inferred inputs
        
        Arguments:
            - fig_width (int) : figure width in inches
            - fig_height (int) : figure height in inches
        '''
        batch_size, steps_size, inputs_size = inputs.shape
    
        figsize = (fig_width, fig_height*len(inputs_size))
        fig, axs = plt.subplots(nrows=len(inputs_size), figsize=figsize)
        fig.suptitle('Input to the generator for a sampled trial', y=1.2)
        inputs = inputs.mean(dim=0).cpu().numpy()
        for jx in range(inputs_size):
            if inputs_size > 1:
                plt.sca(axs[jx])
            else:
                plt.sca(axs)
            plt.plot(time, inputs[:, jx])
            plt.xlabel('time (s)')
        return fig
    
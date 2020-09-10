import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import os

from utils import batchify_random_sample

from conv_lfads import Conv3d_LFADS_Net

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
    def plot_summary(self, model, dl, num_average=200, ix=None, mode='traces', save_dir=None):
        
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
        
        data = dl.dataset.tensors[0]
        
        batch_example, ix = batchify_random_sample(data=data, batch_size=num_average, ix=ix)
        batch_example = batch_example.to(model.device)
        figs_dict['ix'] = ix
        
        model.eval()
        with torch.no_grad():
            if isinstance(model,Conv3d_LFADS_Net):
                recon, (factors, inputs), g_posterior, cout = model(batch_example)
            else:
                recon, (factors, inputs) = model(batch_example)
        
        orig = batch_example[0].cpu().numpy()
#         print(batch_example.shape, data.shape, recon['data'].shape)
        
#         pdb.set_trace()
        
        if mode=='traces':
            figs_dict['traces'] = self.plot_traces(recon['data'].mean(dim=0).detach().cpu().numpy(), orig, mode='activity', norm=True)
            figs_dict['traces'].suptitle('Actual fluorescence trace vs.\nestimated mean for a sampled trial')

        elif mode=='video':
            # TODO
#             figs_dict['videos'] = self.plot_video(recon['data'].mean(dim=0).detach().cpu().numpy(), orig)
            save_video_dir = save_dir + 'videos/'
            if not os.path.exists(save_video_dir):
                os.mkdir(save_video_dir)
            self.plot_video(recon['data'].mean(dim=0).detach().cpu().numpy(), orig, save_folder = save_video_dir)
            # figs_dict['traces'] = self.plot_factors(cout.mean(dim=0).detach().cpu().numpy())

        if self.truth:
            if 'rates' in self.truth.keys():
                recon_rates = recon['rates'].mean(dim=1).cpu().numpy()
                true_rates  = self.truth['rates'][ix]
                figs_dict['truth_rates'] = self.plot_traces(recon_rates, true_rates, mode='rand')
                figs_dict['truth_rates'].suptitle('Reconstructed vs ground-truth rate function')
            
            if 'latent' in self.truth.keys():
                pred_factors = factors.mean(dim=1).cpu().numpy()
                true_factors = self.truth['latent'][ix]
#                 pdb.set_trace()
                figs_dict['truth_factors'] = self.plot_traces(pred_factors, true_factors, num_traces=true_factors.shape[-1], ncols=1)
                figs_dict['truth_factors'].suptitle('Reconstructed vs ground-truth factors')
            else:
                figs_dict['factors'] = self.plot_factors(factors.mean(dim=1).cpu().numpy())
                
            if 'spikes' in self.truth.keys():
                if 'spikes' in recon.keys():
                    recon_spikes = recon['spikes'].mean(dim=1).cpu().numpy()
                    true_spikes  = self.truth['spikes'][ix]
                    figs_dict['truth_spikes'] = self.plot_traces(recon_spikes, true_spikes, mode='rand')
                    figs_dict['truth_spikes'].suptitle('Reconstructed vs ground-truth rate function')
        
        else:
            figs_dict['factors'] = self.plot_factors(factors.mean(dim=1).cpu().numpy())
        
        if inputs is not None:
            figs_dict['inputs'] = self.plot_inputs(inputs.mean(dim=1).cpu().numpy())
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
            if norm is True:
#                 true_norm= (true[:, idx] - np.mean(true[:, idx]))/np.std(true[:, idx])
#                 pred_norm= (pred[:, idx] - np.mean(pred[:, idx]))/np.std(pred[:, idx])
                
#                 plt.sca(ax)
#                 plt.plot(self.time, true_norm, lw=2, color=self.colors['linc_red'])
#                 plt.plot(self.time, pred_norm, lw=2, color=self.colors['linc_blue'])
                plt.sca(ax)
                plt.plot(self.time, true[:, idx], lw=2, color=self.colors['linc_red'])
                plt.plot(self.time, pred[:, idx], lw=2, color=self.colors['linc_blue'])
            
            else:
                plt.sca(ax)
                plt.plot(self.time, true[:, idx], lw=2, color=self.colors['linc_red'])
                plt.plot(self.time, pred[:, idx], lw=2, color=self.colors['linc_blue'])
                
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.legend(['Actual', 'Reconstructed'])
        
        return fig
    
    #------------------------------------------------------------------------------
    def plot_video(self, pred, true, save_folder): #

        num_frames = true.shape[1]
        num_frames_pred = pred.shape[1]
        
        for t in range(num_frames):
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            neg1 = ax1.imshow(pred[0,t,:,:]) 
            neg2 = ax2.imshow(true[0,t,:,:])
            neg1.set_clim(vmin=0, vmax=2)
            neg2.set_clim(vmin=0, vmax=2)
            fig.savefig(save_folder+str(t)+'.png')
            plt.close(fig)
        
        
        
        
    
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
        
        steps_size, factors_size = factors.shape
        
        nrows = min(max_in_col, factors_size)
        ncols = int(np.ceil(factors_size/max_in_col))
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        axs = np.ravel(axs)
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
        steps_size, inputs_size = inputs.shape
    
        figsize = (fig_width, fig_height*inputs_size)
        fig, axs = plt.subplots(nrows=inputs_size, figsize=figsize)
        fig.suptitle('Input to the generator for a sampled trial', y=1.2)
        for jx in range(inputs_size):
            if inputs_size > 1:
                plt.sca(axs[jx])
            else:
                plt.sca(axs)
            plt.plot(self.time, inputs[:, jx])
            plt.xlabel('time (s)')
        return fig
    

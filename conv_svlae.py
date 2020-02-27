import torch
import torch.nn as nn
import torch.nn.functional as F
from lfads import LFADS_Net, LFADS_Encoder, LFADS_ControllerCell
from math import log
import pdb

class Conv1D_SVLAE_Net(nn.Module):
    
    def __init__(self, input_size,
                 deep_g_encoder_size=64, deep_c_encoder_size=64,
                 deep_g_latent_size=32, deep_u_latent_size=1,
                 deep_controller_size=32,
                 obs_encoder_hyperparams = {'sizes' : 5, 'strides'= 1, 'channels': 1, 'layers' : 3}
                 obs_decoder_hyperparams = {'sizes' : 5, 'strides'= 1, 'channels': 1, 'layers' : 3}
                 obs_latent_size=64, obs_controller_size=64,
                 generator_size=64, factor_size=4,
                 prior= {'obs' : {'u'  : {'mean' : {'value': 0.0, 'learnable' : True},
                                          'var'  : {'value': 0.1, 'learnable' : True}}},
                         'deep': {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                          'var'  : {'value': 0.1, 'learnable' : False}},
                                  'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                          'var'  : {'value': 0.1, 'learnable' : True},
                                          'tau'  : {'value': 10,  'learnable' : True}}}},
                 obs_params = {'var'  : {'value' : 0.1, 'learnable' : True}},
                 clip_val = 5.0, dropout=0.0, max_norm=200, generator_burn = 0, 
                 deep_freeze = True, unfreeze = 2000,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        self.input_size = input_size
        self.deep_g_encoder_size = deep_g_encoder_size
        self.deep_c_encoder_size = deep_c_encoder_size
        self.deep_g_latent_size = deep_g_latent_size
        self.deep_u_latent_size = deep_u_latent_size
        self.deep_controller_size = deep_controller_size
        
        self.obs_encoder_hyperparams = obs_encoder_hyperparams
        self.obs_decoder_hyperparams = obs_decoder_hyperparams
        self.obs_latent_size = obs_latent_size
        self.obs_controller_size = obs_controller_size
        
        self.factor_size = factor_size
        self.generator_size = generator_size
        
        self.generator_burn       = generator_burn
        self.clip_val             = clip_val
        self.max_norm             = max_norm
        self.deep_freeze          = deep_freeze
        self.unfreeze             = unfreeze
        self.do_normalize_factors = do_normalize_factors
        self.factor_bias          = factor_bias
        self.device               = device
        
        self.dropout              = torch.nn.Dropout(dropout)
        
        self.obs_model = Conv_Calcium_Net(input_size= self.input_size,
                                          encoder_hyperparams= self.obs_encoder_hyperparams,
                                          decoder_hyperparams= self.obs_decoder_hyperparams,
                                          latent_size = self.obs_latent_size,
                                          controller_size = self.controller_size,
                                          factor_size= self.factor_size,
                                          parameters = self.obs_params,
                                          prior= prior['obs'],
                                          dropout=dropout,
                                          clip_val=self.clip_val,
                                          device=self.device)
        
        self.deep_model           = LFADS_Net(input_size      = self.obs_encoder_size * 2,
                                              g_encoder_size  = self.deep_g_encoder_size,
                                              c_encoder_size  = self.deep_c_encoder_size,
                                              g_latent_size   = self.deep_g_latent_size,
                                              u_latent_size   = self.deep_u_latent_size,
                                              controller_size = self.deep_controller_size,
                                              generator_size  = self.generator_size,
                                              factor_size     = self.factor_size,
                                              prior    = prior['deep'],
                                              clip_val = self.clip_val,
                                              dropout  = dropout,
                                              max_norm = self.max_norm,
                                              do_normalize_factors = self.do_normalize_factors,
                                              factor_bias = self.factor_bias,
                                              device   = self.device)
        
        self.deep_model.add_module('fc_logrates', nn.Linear(self.factor_size, self.input_size))
        
        self.initialize_weights()
        
        if self.deep_freeze:
            for p in self.deep_model.parameters():
                p.requires_grad = False
                
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def initialize_weights(self):
        '''
        initialize_weights()
        
        Initialize weights of network
        '''
                
        with torch.no_grad():
            self.deep_model.initialize_weights()
            self.obs_model.initialize_weights()
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
            
    def unfreeze_parameters(self, step, optimizer, scheduler):
        if step >= self.unfreeze and self.deep_freeze:
            self.deep_freeze = False
            optimizer.add_param_group({'params' : [p for p in self.parameters() if not p.requires_grad],
                                       'lr' : optimizer.param_groups[0]['lr']})
            scheduler.min_lrs.append(scheduler.min_lrs[0])
            for p in self.parameters():
                p.requires_grad_(True)
        return optimizer, scheduler
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def normalize_factors(self):
        self.deep_model.normalize_factors()
    
    
    
class CausalChannelConv1D_Block(nn.Module):
    def __init__(self, )
    
    
#--------
# 1-D CAUSAL CHANNEL-SPECIFIC CONVOLUTION
#--------

class CausalChannelConv1d(torch.nn.Conv2d):
    def __init__(self, kernel_size, out_channels=1, stride=1, dilaton=1, groups=1, bias=True, infer_padding=False):
        
        # Setup padding
        if infer_padding:
            self.__padding = 0
        else:
            self.__padding = (kernel_size - 1)
        
        '''
        CausalChannelConv1d class. Implements channel-specific 1-D causal convolution. Applies same
        set of convolution kernel to every channel independently. Padding only at start of time
        dimension. Output dimensions are same as input dimensions.
        
        __init__(self, kernel_size, bias=True)
        
        required arguments:
            - kernel_size (int) : size of convolution kernel
            - out_channels (int) : number of output channels
            
        optional arguments:
            - bias (bool) : include bias (default=True)
        '''
        super(CausalChannelConv1d, self).__init__(
              in_channels=1,
              out_channels=out_channels,
              kernel_size = (kernel_size, 1),
              stride=1,
              padding=(self.__padding, 0),
              dilation=1,
              groups=1,
              bias=bias)
        
    def forward(self, input):
        # Include false channel dimension
        result = super(CausalChannelConv1d, self).forward(input.unsqueeze(1))
        if self.__padding != 0:
            # Slice tensor to include padding only at start and remove false channel dimension
            return result[:, :, :-self.__padding].squeeze(1)
        else:
            # Remove false channel dimension
            return result.squeeze(1)
    
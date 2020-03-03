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
                 obs_encoder_hyperparams = {'strides'= 1, 'channels': 1, 'layers' : 3}
                 obs_decoder_hyperparams = {'strides'= 1, 'channels': 1, 'layers' : 3}
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
    
    
class Conv_Calcium_Net(nn.Module):
    def __init__(input_size, dense_size,
                 encoder_hyperparams, decoder_hyperparams,
                 latent_size= 64, controller_size= 65, factor_size=4,
                 parameters={'var' : {'value' : 0.1, 'learnable' : True}},
                 prior= {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                 'var'  : {'value': 0.1, 'learnable' : False}},
                         'u'  : {'mean' : {'value': 0.0, 'learnable' : True},
                                 'var'  : {'value': 0.1, 'learnable' : False}}}
                 dropout=0.05, clip_val=5.0, device='cpu'):
        
        super(Conv_Calcium_Net, self).__init__()
        
        self.input_size      = input_size
        self.dense_size      = dense_size
        self.u_latent_size   = latent_size
        self.controller_size = controller_size
        self.factor_size     = factor_size
        self.clip_val        = clip_val
        self.device          = device
        
        self.conv_encoder = make_convnet(encoder_hyperparams)
        
        conv_encoder_out_dim = encoder_hyperparams['channels'][-1] * self.input_size
        self.fc_con= nn.Linear(conv_encoder_out_dim, dense_size)
        
        self.controller   = LFADS_ControllerCell(input_size= conv_encoder_out_dim + self.input_size,
                                                 controller_size= self.controller_size,
                                                 u_latent_size= self.u_latent_size,
                                                 clip_val= self.clip_val,
                                                 dropout= dropout)
        

        self.conv_decoder = make_convnet(decoder_hyperparams)
        
        self.controller_init = nn.Parameter(torch.zeros(self.controller_size))
            
        self.u_prior_mean = torch.ones(self.u_latent_size, device=device) * prior['u']['mean']['value']
        if prior['u']['mean']['learnable']:
            self.u_prior_mean = nn.Parameter(self.u_prior_mean)
        self.u_prior_logvar = torch.ones(self.u_latent_size, device=device) * log(prior['u']['var']['value'])
        if prior['u']['var']['learnable']:
            self.u_prior_logvar = nn.Parameter(self.u_prior_logvar)
            
    def forward():
        pass
        
class CausalRowConv1d_Block(nn.Module):
    def __init__(self, kernel_size, in_channels=1, out_channels=1, stride=1, dilation=1, bias=True, infer_padding=False):
        
        super(CausalRowConv1D_Block, self).__init__()
        
        self.conv  = CausalRowConv1d(kernel_size=kernel_size,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     stride=stride,
                                     dilation=dilation,
                                     bias=bias,
                                     infer_padding=infer_padding)
        
        self.norm   = nn.BatchNorm2d(features=out_channels)
        
        self.nonlin = nn.LeakyRelu()
        
    def forward(self, input):
        return self.nonlin(self.norm(self.conv(input)))
    
    
#--------
# 1-D CAUSAL CHANNEL-SPECIFIC CONVOLUTION
#--------

class CausalRowConv1d(torch.nn.Conv2d):
    def __init__(self, kernel_size, in_channels=1, out_channels=1, stride=1, dilaton=1, groups=1, bias=True, infer_padding=False):
        
        # Setup padding
        if infer_padding:
            self.__padding = 0
        else:
            self.__padding = (kernel_size - 1)
        
        '''
        CausalRowConv1d class. Implements Row-specific 1-D causal convolution. Applies same
        set of convolution kernels to every row independently. Padding only at start of time
        dimension. Output dimensions are same as input dimensions.
        
        __init__(self, kernel_size, bias=True)
        
        required arguments:
            - kernel_size (int) : size of convolution kernel
            - out_channels (int) : number of output channels
            
        optional arguments:
            - bias (bool) : include bias (default=True)
        '''
        super(CausalRowConv1d, self).__init__(
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
        result = super(CausalRowConv1d, self).forward(input)
        if self.__padding != 0:
            # Slice tensor to include padding only at start and remove false channel dimension
            return result[:, :, :-self.__padding]
        else:
            # Remove false channel dimension
            return result
    
    
def make_convnet(conv_params):
    layer_def = zip([1] + conv_params['channels'][1:],
                    conv_params['channels'],
                    conv_params['sizes'],
                    conv_params['strides'],
                    conv_params['dilation'],
                    conv_params['groups'])
    
    
    net = nn.Sequential()
    
    for in_f, out_f, size, stride, dilation, groups in layer_def:
        net.add_module(CausalRowConv1d_Block(kernel_size=size,
                                             in_channels=in_f,
                                             out_channels=out_f,
                                             stride=stride,
                                             dilation=dilation,
                                             groups=groups))
    return net
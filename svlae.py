import torch
import torch.nn as nn
import torch.nn.functional as F
from lfads import LFADS_Net, LFADS_Encoder, LFADS_ControllerCell
from math import log
import pdb

class SVLAE_Net(nn.Module):
    
    def __init__(self, input_size,
                 deep_g_encoder_size= 64, deep_c_encoder_size= 64,
                 obs_encoder_size= 128, obs_latent_size= 64, 
                 deep_g_latent_size= 32, deep_u_latent_size= 1,
                 obs_controller_size= 64, deep_controller_size= 32,
                 generator_size= 64, factor_size= 4,
                 prior= {'obs' : {'u'  : {'mean' : {'value': 0.0, 'learnable' : True},
                                          'var'  : {'value': 0.1, 'learnable' : True}}},
                         'deep': {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                          'var'  : {'value': 0.1, 'learnable' : False}},
                                  'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                          'var'  : {'value': 0.1, 'learnable' : True},
                                          'tau'  : {'value': 10,  'learnable' : True}}}},
                 obs_params = {'gain' : {'value' : 1.0, 'learnable' : False},
                               'bias' : {'value' : 0.0, 'learnable' : False},
                               'tau'  : {'value' : 10., 'learnable' : False},
                               'var'  : {'value' : 0.1, 'learnable' : True}},
                 clip_val = 5.0, dropout=0.0, max_norm=200, generator_burn = 0, 
                 deep_unfreeze_step = 2000, ar1_start_step = 4000,
                 obs_early_stop_step = 2000, obs_continue_step = 8000,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(SVLAE_Net, self).__init__()
        
        self.input_size           = input_size
        self.obs_encoder_size     = obs_encoder_size
        self.obs_latent_size      = obs_latent_size
        self.obs_controller_size  = obs_controller_size
        
        self.deep_g_encoder_size  = deep_g_encoder_size
        self.deep_c_encoder_size  = deep_c_encoder_size
        self.deep_g_latent_size   = deep_g_latent_size
        self.deep_u_latent_size   = deep_u_latent_size
        self.deep_controller_size = deep_controller_size
        
        self.factor_size          = factor_size
        self.generator_size       = generator_size
        
        self.generator_burn       = generator_burn
        self.clip_val             = clip_val
        self.max_norm             = max_norm
        
        self.deep_unfreeze_step   = deep_unfreeze_step
        self.obs_early_stop_step  = obs_early_stop_step
        self.obs_continue_step    = obs_continue_step
        self.ar1_start_step       = ar1_start_step
        
        self.do_normalize_factors = do_normalize_factors
        self.factor_bias          = factor_bias
        
        self.device               = device
        
        self.dropout              = torch.nn.Dropout(dropout)
                
        self.obs_model            = Calcium_Net(input_size      = self.input_size,
                                                encoder_size    = self.obs_encoder_size,
                                                latent_size     = self.obs_latent_size,
                                                controller_size = self.obs_controller_size,
                                                factor_size     = self.factor_size,
                                                parameters      = obs_params,
                                                prior           = prior['obs'],
                                                dropout         = dropout,
                                                clip_val        = self.clip_val,
                                                device          = self.device)
        
        self.deep_model           = LFADS_Net(input_size      = self.input_size,
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
        
        if self.deep_unfreeze_step > 0:
            for p in self.deep_model.parameters():
                p.requires_grad = False
                
        if self.ar1_start_step > 0:
            for p in self.obs_model.generator.calcium_generator.parameters():
                p.requires_grad = False
            self.obs_model.generator.calcium_generator.logvar.requires_grad = True
            
    def forward(self, input):
        
        input = input.permute(1, 0, 2)
        self.steps_size, self.batch_size, input_size = input.shape
        assert input_size == self.input_size, 'input_size does not match self.input_size'
        
        obs_encoder_state, obs_controller_state = self.obs_model.initialize_hidden_states(input)
        
        out_obs_enc = self.obs_model.encoder(input, obs_encoder_state)

        input_deep =  input
        deep_g_encoder_state, deep_c_encoder_state, deep_controller_state = self.deep_model.initialize_hidden_states(input_deep)
        
        self.deep_model.g_posterior_mean, self.deep_model.g_posterior_logvar, out_deep_g_enc, out_deep_c_enc = self.deep_model.encoder(input_deep, (deep_g_encoder_state, deep_c_encoder_state))
        
        deep_generator_state = self.deep_model.fc_genstate(self.deep_model.sample_gaussian(self.deep_model.g_posterior_mean, self.deep_model.g_posterior_logvar))
        
        factor_state = self.deep_model.generator.fc_factors(self.deep_model.dropout(deep_generator_state))
        
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        obs_state = torch.zeros(self.batch_size, self.input_size, device=self.device)
        spike_state = torch.zeros(self.batch_size, self.input_size, device=self.device)
        
        spikes = torch.empty(0, self.batch_size, self.input_size, device=self.device)
        obs    = torch.empty(0, self.batch_size, self.input_size, device=self.device)
        
        self.obs_model.u_posterior_mean   = torch.empty(self.batch_size, 0, self.obs_latent_size, device=self.device)
        self.obs_model.u_posterior_logvar = torch.empty(self.batch_size, 0, self.obs_latent_size, device=self.device)
        
        if self.deep_c_encoder_size > 0 and self.deep_controller_size > 0 and self.deep_u_latent_size > 0:
            deep_gen_inputs = torch.empty(0, self.batch_size, self.deep_u_latent_size, device=self.device)
            
            # initialize u posterior store
            self.deep_model.u_posterior_mean   = torch.empty(self.batch_size, 0, self.deep_u_latent_size, device=self.device)
            self.deep_model.u_posterior_logvar = torch.empty(self.batch_size, 0, self.deep_u_latent_size, device=self.device)
        
        for t in range(self.generator_burn):
            deep_generator_state, factor_state = self.deep_model.generator(None, deep_generator_state)
        
        for t in range(self.steps_size):
            
            if self.deep_c_encoder_size > 0 and self.deep_controller_size > 0 and self.deep_u_latent_size > 0:

                deep_u_mean, deep_u_logvar, deep_controller_state = self.deep_model.controller(torch.cat((out_deep_c_enc[t], factor_state), dim=1), deep_controller_state)

                self.deep_model.u_posterior_mean = torch.cat((self.deep_model.u_posterior_mean, deep_u_mean.unsqueeze(0)), dim=0)
                self.deep_model.u_posterior_logvar = torch.cat((self.deep_model.u_posterior_logvar, deep_u_logvar.unsqueeze(0)), dim=0)
                deep_generator_input = self.deep_model.sample_gaussian(self.deep_model.u_posterior_mean, self.deep_model.u_posterior_logvar)
                deep_gen_inputs = torch.cat((deep_gen_inputs, deep_generator_input.unsqueeze(0)), dim=0)
            else:
                deep_generator_input = torch.empty(self.batch_size, self.deep_u_latent_size, device=self.device)
                deep_gen_inputs = None
            
            obs_u_mean, obs_u_logvar, obs_controller_state = self.obs_model.controller(torch.cat((out_obs_enc[t], obs_state), dim=1), obs_controller_state)
#             pdb.set_trace()
            self.obs_model.u_posterior_mean   = torch.cat((self.obs_model.u_posterior_mean, obs_u_mean.unsqueeze(1)), dim=1)
            self.obs_model.u_posterior_logvar = torch.cat((self.obs_model.u_posterior_logvar, obs_u_logvar.unsqueeze(1)), dim=1)
            
            obs_generator_state = self.obs_model.sample_gaussian(obs_u_mean, obs_u_logvar)
            
            deep_generator_state, factor_state = self.deep_model.generator(deep_generator_input, deep_generator_state)
            
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            
            obs_state, spike_state = self.obs_model.generator(torch.cat((obs_generator_state, factor_state), dim=1), obs_state)
            
#             pdb.set_trace()
            obs = torch.cat((obs, obs_state.unsqueeze(0)), dim=0)
            spikes = torch.cat((spikes, spike_state.unsqueeze(0)), dim=0)
            
        if self.deep_c_encoder_size > 0 and self.deep_controller_size > 0 and self.deep_u_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.deep_model.u_prior_mean, self.deep_model.u_prior_logvar = self.deep_model._gp_to_normal(self.deep_model.u_prior_gp_mean, self.deep_model.u_prior_gp_logvar, self.deep_model.u_prior_gp_logtau, deep_gen_inputs)
            
        recon = {}
        recon['rates'] = self.deep_model.fc_logrates(factors).exp()
        recon['data']  = obs.permute(1, 0, 2)
        recon['spikes'] = spikes
        
        return recon, (factors, deep_gen_inputs)
    


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
    
    def change_parameter_grad_status(self, step, optimizer, scheduler, loading_checkpoint=False):
        
        def step_condition(run_step, status_step, loading_checkpoint):
            if loading_checkpoint:
                return run_step >= status_step
            else:
                return run_step == status_steps
        
        if step_condition(step, self.deep_unfreeze_step, loading_checkpoint):
            print('Unfreezing deep model parameters', flush=True)
            optimizer.add_param_group({'params' : [p for p in self.deep_model.parameters() if not p.requires_grad],
                                       'lr' : optimizer.param_groups[0]['lr']})
            scheduler.min_lrs.append(scheduler.min_lrs[0])
            for p in self.deep_model.parameters():
                p.requires_grad_(True)
                
        if step_condition(step, self.obs_early_stop_step, loading_checkpoint):
            print('Stopping observation model parameters', flush=True)
            del optimizer.param_groups[0]
            del scheduler.min_lrs[0]
            for p in self.obs_model.parameters():
                p.requires_grad_(False)
                
        if step_condition(step, self.obs_continue_step, loading_checkpoint):
            print('Continuing observation model parameters', flush=True)
            optimizer.add_param_group({'params' : [p for p in self.obs_model.parameters() if not p.requires_grad],
                                       'lr' : optimizer.param_groups[0]['lr']})
            scheduler.min_lrs.append(scheduler.min_lrs[0])
            for p in self.obs_model.parameters():
                p.requires_grad_(True)
                
        if step_condition(step, self.ar1_start_step, loading_checkpoint):
            print('Starting AR1 model parameters', flush=True)
            optimizer.add_param_group({'params' : [p for p in self.obs_model.generator.calcium_generator.parameters() if not p.requires_grad],
                                       'lr' : optimizer.param_groups[0]['lr']})
            scheduler.min_lrs.append(scheduler.min_lrs[0])
            for p in self.obs_model.generator.calcium_generator.parameters():
                p.requires_grad_(True)
        
        return optimizer, scheduler
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def normalize_factors(self):
        self.deep_model.normalize_factors()
        

class Calcium_Net(nn.Module):
    def __init__(self, input_size, encoder_size=128,
                 latent_size=64, controller_size=128, factor_size=4,
                 parameters = {'gain' : {'value' : 1.0, 'learnable' : False},
                               'bias' : {'value' : 0.0, 'learnable' : False},
                               'tau'  : {'value' : 10, 'learnable' : False},
                               'var' :  {'value' : 0.1, 'learnable' : True}},
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}}},
                 clip_val = 5.0, dropout=0.05, device='cpu'):
        
        super(Calcium_Net, self).__init__()
        
        self.input_size      = input_size
        self.encoder_size    = encoder_size
        self.u_latent_size   = latent_size
        self.controller_size = controller_size
        self.factor_size     = factor_size
        self.clip_val        = clip_val
        self.device          = device

        self.encoder         = Calcium_Encoder(input_size= self.input_size,
                                               encoder_size= self.encoder_size,
                                               clip_val= self.clip_val,
                                               dropout= dropout)

        
        self.controller      = LFADS_ControllerCell(input_size      = self.encoder_size*2 + self.input_size,
                                                         controller_size = self.controller_size,
                                                         u_latent_size   = self.u_latent_size,
                                                         clip_val        = self.clip_val,
                                                         dropout         = dropout)
        
        self.generator       = Calcium_Generator(input_size  = self.u_latent_size + self.factor_size,
                                                 output_size = self.input_size,
                                                 parameters  = parameters,
                                                 dropout     = dropout,
                                                 device      = self.device)
        
        # Initialize learnable biases
        self.encoder_init    = nn.Parameter(torch.zeros(2, self.encoder_size))
        self.controller_init = nn.Parameter(torch.zeros(self.controller_size))
            
        self.u_prior_mean = torch.ones(self.u_latent_size, device=device) * prior['u']['mean']['value']
        if prior['u']['mean']['learnable']:
            self.u_prior_mean = nn.Parameter(self.u_prior_mean)
        self.u_prior_logvar = torch.ones(self.u_latent_size, device=device) * log(prior['u']['var']['value'])
        if prior['u']['var']['learnable']:
            self.u_prior_logvar = nn.Parameter(self.u_prior_logvar)
            
    def forward():
        pass
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def initialize_weights(self):
        '''
        initialize_weights()
        
        Initialize weights of network
        '''
        
        def standard_init(weights):
            k = weights.shape[1] # dimensionality of inputs
            weights.data.normal_(std=k**-0.5) # inplace resetting W ~ N(0, 1/sqrt(K))
        
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'weight' in name:
                    standard_init(p)
                    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    
    def sample_gaussian(self, mean, logvar):
        '''
        sample_gaussian(mean, logvar)
        
        Sample from a diagonal gaussian with given mean and log-variance
        
        Required Arguments:
            - mean (torch.Tensor)   : mean of diagional gaussian
            - logvar (torch.Tensor) : log-variance of diagonal gaussian
        '''
        # Generate noise from standard gaussian
        eps = torch.randn(mean.shape, requires_grad=False).to(self.device)
        # Scale and shift by mean and standard deviation
        return torch.exp(logvar*0.5)*eps + mean
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    
    def initialize_hidden_states(self, input):
        '''
        initialize_hidden_states()
        
        Initialize hidden states of recurrent networks
        '''
        
        self.steps_size, self.batch_size, input_size = input.shape
        assert input_size == self.input_size, 'Input is expected to have dimensions [%i, %i, %i]'%(self.steps_size, self.batch_size, self.input_size)
        
        encoder_state  = (torch.ones(self.batch_size, 2,  self.encoder_size, device=self.device) * self.encoder_init).permute(1, 0, 2)
        controller_state = torch.ones(self.batch_size, self.controller_size, device=self.device) * self.controller_init
        return encoder_state, controller_state
            
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
class Calcium_Encoder(nn.Module):
    '''
    Calcium_Encoder
    
    Calcium Encoder Network 
    
    __init__(self, input_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0)
    
    Required Arguments:
        - input_size (int):  size of input dimensions
        - encoder_size (int):  size of generator encoder network
        
    Optional Arguments:
        - dropout (float): dropout probability
        - clip_val (float): RNN hidden state value limit
        
    '''
    def __init__(self, input_size, encoder_size, dropout= 0.0, clip_val= 5.0):
        super(Calcium_Encoder, self).__init__()
        self.input_size      = input_size
        self.encoder_size  = encoder_size
        self.clip_val        = clip_val
        self.dropout = nn.Dropout(dropout)        
        
        # encoder BiRNN
        self.gru  = nn.GRU(input_size=self.input_size, hidden_size=self.encoder_size, bidirectional=True)
            
    def forward(self, input, hidden):
        encoder_init = hidden
        
        # Run bidirectional RNN over data
        out_gru, hidden_gru = self.gru(self.dropout(input), encoder_init.contiguous())
        out_gru = out_gru.clamp(min=-self.clip_val, max=self.clip_val)
        
        return out_gru
        
class Calcium_Generator(nn.Module):
    def __init__(self, input_size, output_size, parameters, dropout, device='cpu'):
        super(Calcium_Generator, self).__init__()
        
        self.input_size  = input_size
        self.output_size = output_size
        self.device      = device
        
        self.spike_generator = Spike_Generator(input_size=input_size, output_size=output_size, dropout=dropout, device=device)
        self.calcium_generator = AR1_Calcium(parameters=parameters, device=device)
        
    def forward(self, input, hidden):
        calcium_state = hidden
        spike_state = self.spike_generator(input)
        calcium_state = self.calcium_generator(spike_state, calcium_state)
        return calcium_state, spike_state

class Spike_Generator(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0, device='cpu'):
        super(Spike_Generator, self).__init__()
    
        self.fc_logspike = nn.Linear(in_features=input_size, out_features=output_size)
        self.dropout     = nn.Dropout(dropout)
        self.device      = device
    
    def forward(self, input):
#         pdb.set_trace()
        return torch.clamp(self.fc_logspike(self.dropout(input)).exp() - 1, min=0.0)
    
class AR1_Calcium(nn.Module):
    
    def __init__(self, parameters = {'gain': {'value' : 1.0, 'learnable': False},
                                     'bias': {'value' : 0.0, 'learnable': False},
                                     'tau' : {'value' : 10,  'learnable': False},
                                     'var' : {'value' : 0.1, 'learnable': True}},
                 device = 'cpu'):
        
        super(AR1_Calcium, self).__init__()
        
        self.device= device
        
        self.gain   = nn.Parameter(torch.tensor(parameters['gain']['value'], device=device, dtype=torch.float32)) if parameters['gain'] ['learnable'] else torch.tensor(parameters['gain']['value'], device=device, dtype=torch.float32)
        self.bias   = nn.Parameter(torch.tensor(parameters['bias']['value'], device=device, dtype=torch.float32)) if parameters['bias']['learnable'] else torch.tensor(parameters['bias']['value'], device=device, dtype=torch.float32)
        self.logtau = nn.Parameter(torch.tensor(log(parameters['tau']['value']), device=device, dtype=torch.float32)) if parameters['tau']['learnable'] else torch.tensor(log(parameters['tau']['value']), device=device, dtype=torch.float32)
        self.logvar = nn.Parameter(torch.tensor(log(parameters['var']['value']), device=device, dtype=torch.float32)) if parameters['var']['learnable'] else torch.tensor(log(parameters['var']['value']), device=device, dtype=torch.float32)

    def forward(self, input, hidden):
#         pdb.set_trace()
        return hidden * (1.0-1.0/self.logtau.exp()) + self.gain * input + self.bias
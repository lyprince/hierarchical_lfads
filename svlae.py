import torch
import torch.nn as nn
import torch.nn.functional as F
from lfads import LFADS_Net, LFADS_Encoder, LFADS_ControllerCell
from math import log
import pdb

class SVLAE_Net(nn.Module):
    
    def __init__(self, input_size,
                 g1_encoder_size= 128, g2_encoder_size= 64,
                 c1_encoder_size= 128, c2_encoder_size= 64,
                 g1_latent_size= 64, g2_latent_size= 32,
                 u1_latent_size= 64, u2_latent_size= 1,
                 controller1_size= 64, controller2_size= 32,
                 generator_size= 64, factor_size= 4,
                 prior= {'g0_1' : {'mean' : {'value': 0.0, 'learnable' : True},
                                   'var'  : {'value': 0.1, 'learnable' : False}},
                         'g0_2' : {'mean' : {'value': 0.0, 'learnable' : True},
                                   'var'  : {'value': 0.1, 'learnable' : False}},
                         'u_1' : {'mean' : {'value': 0.0, 'learnable' : True},
                                   'var'  : {'value': 0.1, 'learnable' : False}},
                         'u_2'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                   'var'  : {'value': 0.1, 'learnable' : True},
                                   'tau'  : {'value': 10,  'learnable' : True}}},
                 obs_params = {'gain' : {'value' : 1.0, 'learnable' : False},
                               'bias' : {'value' : 0.0, 'learnable' : False},
                               'tau'  : {'value' : 10, 'learnable' : False},
                               'var' :  {'value' : 0.1, 'learnable' : True}},
                 clip_val = 5.0, dropout=0.0, max_norm=200,
                 deep_freeze = True, unfreeze = 2000,
                 do_normalize_factors=True, device='cpu'):
    
        super(SVLAE_Net, self).__init__()
        
        self.input_size           = input_size
        self.g1_encoder_size      = g1_encoder_size
        self.c1_encoder_size      = c1_encoder_size
        self.g1_latent_size       = g1_latent_size
        self.u1_latent_size       = u1_latent_size
        self.controller1_size     = controller1_size
        
        self.g2_encoder_size      = g2_encoder_size
        self.c2_encoder_size      = c2_encoder_size
        self.g2_latent_size       = g2_latent_size
        self.u2_latent_size       = u2_latent_size
        self.controller2_size     = controller2_size
        
        self.factor_size          = factor_size
        self.generator_size       = generator_size
        
        self.clip_val             = clip_val
        self.max_norm             = max_norm
        self.deep_freeze          = deep_freeze
        self.unfreeze             = unfreeze
        self.do_normalize_factors = do_normalize_factors
        self.device               = device
        
        self.dropout              = torch.nn.Dropout(dropout)
        
        self.encoder              = LFADS_Encoder(input_size     = self.input_size,
                                                  g_encoder_size = self.g1_encoder_size,
                                                  c_encoder_size = self.c1_encoder_size,
                                                  g_latent_size  = self.g1_latent_size,
                                                  clip_val       = self.clip_val,
                                                  dropout        = dropout)
        
        self.controller           = LFADS_ControllerCell(input_size      = self.c1_encoder_size*2 + self.input_size,
                                                         controller_size = self.controller1_size,
                                                         u_latent_size   = self.u1_latent_size,
                                                         clip_val        = self.clip_val,
                                                         dropout         = dropout)
        
        self.obs_model            = Calcium_Generator(input_size  = self.u1_latent_size + self.factor_size,
                                                      output_size = self.input_size,
                                                      parameters  = obs_params,
                                                      dropout     = dropout,
                                                      device      = self.device)
        
        self.deep_model           = LFADS_Net(input_size      = self.g1_encoder_size * 2,
                                              output_size     = self.input_size,
                                              g_encoder_size  = self.g2_encoder_size,
                                              c_encoder_size  = self.c2_encoder_size,
                                              g_latent_size   = self.g2_latent_size,
                                              u_latent_size   = self.u2_latent_size,
                                              controller_size = self.controller2_size,
                                              generator_size  = self.generator_size,
                                              factor_size     = self.factor_size,
                                              prior    = prior['deep'],
                                              clip_val = self.clip_val,
                                              dropout  = dropout,
                                              max_norm = self.max_norm,
                                              do_normalize_factors = self.do_normalize_factors,
                                              device   = self.device)
        
        # Initialize learnable biases
        self.g_encoder_init  = nn.Parameter(torch.zeros(2, self.g1_encoder_size))
        self.c_encoder_init  = nn.Parameter(torch.zeros(2, self.c1_encoder_size))
        self.controller_init = nn.Parameter(torch.zeros(self.controller1_size))
        
        self.g_prior_mean = torch.ones(self.g1_latent_size, device=device) * prior['obs']['g0']['mean']['value']
        if prior['obs']['g0']['mean']['learnable']:
            self.g_prior_mean = nn.Parameter(self.g_prior_mean)
        self.g_prior_logvar = torch.ones(self.g1_latent_size, device=device) * log(prior['obs']['g0']['var']['value'])
        if prior['obs']['g0']['var']['learnable']:
            self.g_prior_logvar = nn.Parameter(self.g_prior_logvar)
            
        self.u_prior_mean = torch.ones(self.u1_latent_size, device=device) * prior['obs']['u']['mean']['value']
        if prior['obs']['u']['mean']['learnable']:
            self.u_prior_mean = nn.Parameter(self.u_prior_mean)
        self.u_prior_logvar = torch.ones(self.u1_latent_size, device=device) * log(prior['obs']['u']['var']['value'])
        if prior['obs']['u']['var']['learnable']:
            self.u_prior_logvar = nn.Parameter(self.u_prior_logvar)
        
        self.initialize_weights()
        
        if self.deep_freeze:
            for p in self.deep_model.parameters():
                p.requires_grad = False
            
    def forward(self, input):
        
        g1_encoder_state, c1_encoder_state, controller1_state = self.initialize_hidden_states(input)
        
        self.g_posterior_mean, self.g_posterior_logvar, out_gru_g1_enc, out_gru_c1_enc = self.encoder(input, (g1_encoder_state, c1_encoder_state))
        
        generator1_state = self.sample_gaussian(self.g_posterior_mean, self.g_posterior_logvar)
#         pdb.set_trace()
        
        g2_encoder_state, c2_encoder_state, controller2_state = self.deep_model.initialize_hidden_states(out_gru_g1_enc)
        
        self.deep_model.g_posterior_mean, self.deep_model.g_posterior_logvar, out_gru_g2_enc, out_gru_c2_enc = self.deep_model.encoder(out_gru_g1_enc, (g2_encoder_state, c2_encoder_state))
        
        generator2_state = self.deep_model.fc_genstate(self.deep_model.sample_gaussian(self.deep_model.g_posterior_mean, self.deep_model.g_posterior_logvar))
        
        factor_state = self.deep_model.generator.fc_factors(self.deep_model.dropout(generator2_state))
        
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        obs_state = torch.zeros(self.batch_size, self.input_size, device=self.device)
        obs_state, spike_state = self.obs_model(torch.cat((generator1_state, factor_state), dim=1), obs_state)
        
        spikes = torch.empty(0, self.batch_size, self.input_size, device=self.device)
        obs    = torch.empty(0, self.batch_size, self.input_size, device=self.device)
        
        self.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u1_latent_size, device=self.device)
        self.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u1_latent_size, device=self.device)
        
        if self.c2_encoder_size > 0 and self.controller2_size > 0 and self.u2_latent_size > 0:
            gen2_inputs = torch.empty(0, self.batch_size, self.u2_latent_size, device=self.device)
            
            # initialize u posterior store
            self.deep_model.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u2_latent_size, device=self.device)
            self.deep_model.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u2_latent_size, device=self.device)
        
        for t in range(self.steps_size):
            
            if self.c2_encoder_size > 0 and self.controller2_size > 0 and self.u2_latent_size > 0:

                u2_mean, u2_logvar, controller2_state = self.deep_model.controller(torch.cat((out_gru_c2_enc[t], factor_state), dim=1), controller2_state)

                self.deep_model.u_posterior_mean = torch.cat((self.deep_model.u_posterior_mean, u2_mean.unsqueeze(0)), dim=0)
                self.deep_model.u_posterior_logvar = torch.cat((self.deep_model.u_posterior_logvar, u2_logvar.unsqueeze(0)), dim=0)
                generator2_input = self.deep_model.sample_gaussian(self.deep_model.u_posterior_mean, self.deep_model.u_posterior_logvar)
                gen2_inputs = torch.cat((gen2_inputs, generator2_input.unsqueeze(0)), dim=0)
            
            else:
                generator2_input = torch.empty(self.batch_size, self.u2_latent_size, device=self.device)
                gen2_inputs = None
            
            u1_mean, u1_logvar, controller1_state = self.controller(torch.cat((out_gru_c1_enc[t], spike_state), dim=1), controller1_state)
#             pdb.set_trace()
            self.u_posterior_mean   = torch.cat((self.u_posterior_mean, u1_mean.unsqueeze(1)), dim=1)
            self.u_posterior_logvar = torch.cat((self.u_posterior_logvar, u1_logvar.unsqueeze(1)), dim=1)
            
            generator1_state = self.sample_gaussian(u1_mean, u1_logvar)
            
            generator2_state, factor_state = self.deep_model.generator(generator2_input, generator2_state)
            
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            
            obs_state, spike_state = self.obs_model(torch.cat((generator1_state, factor_state), dim=1), obs_state)
            
#             pdb.set_trace()
            obs = torch.cat((obs, obs_state.unsqueeze(0)), dim=0)
            spikes = torch.cat((spikes, spike_state.unsqueeze(0)), dim=0)
            
        if self.c2_encoder_size > 0 and self.controller2_size > 0 and self.u2_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.u_prior_mean, self.u_prior_logvar = self._gp_to_normal(self.u_prior_gp_mean, self.u_prior_gp_logvar, self.u_prior_gp_logtau, gen_inputs)
            
        recon = {}
        recon['rates'] = self.deep_model.fc_rates(factors).exp()
        recon['data']  = obs
        recon['spikes'] = spikes
        
        return recon, (factors, gen2_inputs)
    
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
        
        g_encoder_state  = (torch.ones(self.batch_size, 2,  self.g1_encoder_size, device=self.device) * self.g_encoder_init).permute(1, 0, 2)
        c_encoder_state  = (torch.ones(self.batch_size, 2,  self.c1_encoder_size, device=self.device) * self.c_encoder_init).permute(1, 0, 2)
        controller_state = torch.ones(self.batch_size, self.controller1_size, device=self.device) * self.controller_init
        return g_encoder_state, c_encoder_state, controller_state

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

            if self.do_normalize_factors:
                self.deep_model.normalize_factors()
    
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
        

class Calcium_Net(nn.Module):
    def __init__(self, input_size,
                 g_encoder_size=128, c_encoder_size=128,
                 g_latent_size=64, u_latent_size=64,
                 controller_size=128, factor_size=4,
                 parameters = {'gain' : {'value' : 1.0, 'learnable' : False},
                               'bias' : {'value' : 0.0, 'learnable' : False},
                               'tau'  : {'value' : 10, 'learnable' : False},
                               'var' :  {'value' : 0.1, 'learnable' : True}},
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}}},
                 clip_val = 5.0, dropout=0.05, device='cpu'):
        
        self.input_size      = input_size
        self.g_encoder_size  = g_encoder_size
        self.c_encoder_size  = c_encoder_size
        self.g_latent_size   = g_latent_size
        self.u_latent_size   = u_latent_size
        self.controller_size = controller_size
        self.factor_size     = factor_size

    
        self.encoder              = LFADS_Encoder(input_size     = self.input_size,
                                                  g_encoder_size = self.g_encoder_size,
                                                  c_encoder_size = self.c_encoder_size,
                                                  g_latent_size  = self.g_latent_size,
                                                  clip_val       = self.clip_val,
                                                  dropout        = dropout)
        
        self.controller           = LFADS_ControllerCell(input_size      = self.c_encoder_size*2 + self.input_size,
                                                         controller_size = self.controller_size,
                                                         u_latent_size   = self.u_latent_size,
                                                         clip_val        = self.clip_val,
                                                         dropout         = dropout)
        
        self.generator            = Calcium_Generator(input_size  = self.u1_latent_size + self.factor_size,
                                                      output_size = self.input_size,
                                                      parameters  = parameters,
                                                      dropout     = dropout,
                                                      device      = self.device)
        
        # Initialize learnable biases
        self.g_encoder_init  = nn.Parameter(torch.zeros(2, self.g1_encoder_size))
        self.c_encoder_init  = nn.Parameter(torch.zeros(2, self.c1_encoder_size))
        self.controller_init = nn.Parameter(torch.zeros(self.controller1_size))
        
        self.g_prior_mean = torch.ones(self.g1_latent_size, device=device) * prior['g0']['mean']['value']
        if prior['g0']['mean']['learnable']:
            self.g_prior_mean = nn.Parameter(self.g_prior_mean)
        self.g_prior_logvar = torch.ones(self.g1_latent_size, device=device) * log(prior['g0']['var']['value'])
        if prior['g0']['var']['learnable']:
            self.g_prior_logvar = nn.Parameter(self.g_prior_logvar)
            
        self.u_prior_mean = torch.ones(self.u1_latent_size, device=device) * prior['u']['mean']['value']
        if prior['u']['mean']['learnable']:
            self.u_prior_mean = nn.Parameter(self.u_prior_mean)
        self.u_prior_logvar = torch.ones(self.u1_latent_size, device=device) * log(prior['u']['var']['value'])
        if prior['u']['var']['learnable']:
            self.u_prior_logvar = nn.Parameter(self.u_prior_logvar)
        
        self.initialize_weights()
        
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
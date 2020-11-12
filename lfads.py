import torch
import torch.nn as nn
import torch.nn.functional as F
from objective import kldiv_gaussian_gaussian
from rnn import LFADS_GenGRUCell
from math import log
import pdb

class LFADS_Net(nn.Module):
    '''
    LFADS_Net (Latent Factor Analysis via Dynamical Systems) neural network class.
    
    __init__(self, input_size, factor_size = 4,
                   g_encoder_size = 64, c_encoder_size = 64,
                   g_latent_size = 64, u_latent_size = 1,
                   controller_size= 64, generator_size = 64,
                   prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                    'var'  : {'value': 0.1, 'learnable' : False}},
                             'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                    'var'  : {'value': 0.1, 'learnable' : True},
                                     'tau'  : {'value': 10,  'learnable' : True}}},
                   clip_val=5.0, dropout=0.0, max_norm = 200,
                   do_normalize_factors=True, device='cpu')
                   
    Required Arguments:
        - input_size (int) : size of input dimensions (number of cells)
    Optional Arguments:
        - g_encoder_size     (int): size of generator encoder network
        - c_encoder_size     (int): size of controller encoder network
        - g_latent_size      (int): size of generator ic latent variable
        - u_latent_size      (int): size of generator input latent variable
        - controller_size    (int): size of controller network
        - generator_size     (int): size of generator network
        - prior             (dict): dictionary of prior distribution parameters
        - clip_val         (float): RNN hidden state value limit
        - dropout          (float): dropout probability
        - max_norm           (int): maximum gradient norm
        - do_normalize_factors (bool): whether to normalize factors
        - device          (string): device to use
    '''
    
    def __init__(self, input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(LFADS_Net, self).__init__()
        
        self.input_size           = input_size
#         self.output_size          = input_size if output_size is None else output_size
        self.g_encoder_size       = g_encoder_size
        self.c_encoder_size       = c_encoder_size
        self.g_latent_size        = g_latent_size
        self.u_latent_size        = u_latent_size
        self.controller_size      = controller_size
        self.generator_size       = generator_size
        self.factor_size          = factor_size
        
        self.clip_val             = clip_val
        self.max_norm             = max_norm
        self.do_normalize_factors = do_normalize_factors
        self.factor_bias          = factor_bias
        self.device               = device
        self.deep_freeze          = deep_freeze
        
        self.dropout              = torch.nn.Dropout(dropout)

        # Initialize encoder RNN
        self.encoder     = LFADS_Encoder(input_size     = self.input_size,
                                         g_encoder_size = self.g_encoder_size,
                                         c_encoder_size = self.c_encoder_size,
                                         g_latent_size  = self.g_latent_size,
                                         clip_val       = self.clip_val,
                                         dropout        = dropout)
        
        # Initialize controller RNN
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.controller  = LFADS_ControllerCell(input_size      = self.c_encoder_size*2 + self.factor_size,
                                                    controller_size = self.controller_size,
                                                    u_latent_size   = self.u_latent_size,
                                                    clip_val        = self.clip_val,
                                                    dropout         = dropout)
        
        # Initialize generator RNN
        self.generator   = LFADS_GeneratorCell(input_size     = self.u_latent_size,
                                               generator_size = self.generator_size,
                                               factor_size    = self.factor_size,
                                               clip_val       = self.clip_val,
                                               factor_bias    = self.factor_bias,
                                               dropout        = dropout)
        
        # Initialize dense layers
        if self.g_latent_size == self.generator_size:
            self.fc_genstate = Identity(in_features=self.g_latent_size, out_features=self.generator_size)
        else:
            self.fc_genstate = nn.Linear(in_features= self.g_latent_size, out_features= self.generator_size)
                
        # Initialize learnable biases
        self.g_encoder_init  = nn.Parameter(torch.zeros(2, self.g_encoder_size))
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.c_encoder_init  = nn.Parameter(torch.zeros(2, self.c_encoder_size))
            self.controller_init = nn.Parameter(torch.zeros(self.controller_size))
        
        # Initialize priors
#         self.register_buffer('g_prior_mean',None)
#         self.register_buffer('g_prior_logvar',None)
#         self.register_buffer('g_posterior_mean',None)
#         self.register_buffer('g_posterior_logvar',None)
        
        self.g_prior_mean = torch.ones(self.g_latent_size, device=device) * prior['g0']['mean']['value']
        
        if prior['g0']['mean']['learnable']:
            self.g_prior_mean = nn.Parameter(self.g_prior_mean)
        self.g_prior_logvar = torch.ones(self.g_latent_size, device=device) * log(prior['g0']['var']['value'])
        if prior['g0']['var']['learnable']:
            self.g_prior_logvar = nn.Parameter(self.g_prior_logvar)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            self.u_prior_gp_mean = torch.ones(self.u_latent_size, device=device) * prior['u']['mean']['value']
            if prior['u']['mean']['learnable']:
                self.u_prior_gp_mean = nn.Parameter(self.u_prior_gp_mean)
            self.u_prior_gp_logvar = torch.ones(self.u_latent_size, device=device) * log(prior['u']['var']['value'])
            if prior['u']['var']['learnable']:
                self.u_prior_gp_logvar = nn.Parameter(self.u_prior_gp_logvar)
            self.u_prior_gp_logtau = torch.ones(self.u_latent_size, device=device) * log(prior['u']['tau']['value'])
            if prior['u']['tau']['learnable']:
                self.u_prior_gp_logtau = nn.Parameter(self.u_prior_gp_logtau)
        
        # Initialize weights
        self.initialize_weights()
        
    def forward(self, input):
        '''
        forward(input)
        
        Required Arguments:
            - input (torch.Tensor): input data with dimensions [time x batch x cells]
        '''
        import time
        tic = time.time()

        # Initialize hidden states
        g_encoder_state, c_encoder_state, controller_state = self.initialize_hidden_states(input) 

        
        # Encode input and calculate and calculate generator initial condition variational posterior distribution
        self.g_posterior_mean, self.g_posterior_logvar, out_gru_g_enc, out_gru_c_enc = self.encoder(input, (g_encoder_state, c_encoder_state))
        

        # Sample generator state
        generator_state = self.fc_genstate(self.sample_gaussian(self.g_posterior_mean, self.g_posterior_logvar))
        
        # Initialize factor state
        factor_state = self.generator.fc_factors(self.dropout(generator_state))
        
        # Factors store
        factors = torch.empty(0, self.batch_size, self.factor_size, device=self.device)
        
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # initialize generator input store
            gen_inputs = torch.empty(0, self.batch_size, self.u_latent_size, device=self.device)
            
            # initialize u posterior store
            self.u_posterior_mean   = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
            self.u_posterior_logvar = torch.empty(self.batch_size, 0, self.u_latent_size, device=self.device)
        
        tic = time.time()
        
        # Controller and Generator Loop
        for t in range(self.steps_size):
            if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
                # Update controller state and calculate generator input variational posterior distribution
                u_mean, u_logvar, controller_state = self.controller(torch.cat((out_gru_c_enc[t], factor_state), dim=1), controller_state)
                
                # Append u_posterior mean and logvar
                self.u_posterior_mean = torch.cat((self.u_posterior_mean, u_mean.unsqueeze(1)), dim=1)
                self.u_posterior_logvar = torch.cat((self.u_posterior_logvar, u_logvar.unsqueeze(1)), dim=1)

                # Sample generator input
                generator_input = self.sample_gaussian(u_mean, u_logvar)
                # Append generator input to store
                gen_inputs  = torch.cat((gen_inputs, generator_input.unsqueeze(0)), dim=0)
            else:
                generator_input = torch.empty(self.batch_size, self.u_latent_size, device=self.device)
                gen_inputs = None
                
            # Update generator and factor state
            generator_state, factor_state = self.generator(generator_input, generator_state)
            # Store factor state
            factors = torch.cat((factors, factor_state.unsqueeze(0)), dim=0)
            
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            # Instantiate AR1 process as mean and variance per time step
            self.u_prior_mean, self.u_prior_logvar = self._gp_to_normal(self.u_prior_gp_mean, self.u_prior_gp_logvar, self.u_prior_gp_logtau, gen_inputs)
        
        return (factors, gen_inputs)
    
    def sample_gaussian(self, mean, logvar):
        '''
        sample_gaussian(mean, logvar)
        
        Sample from a diagonal gaussian with given mean and log-variance
        
        Required Arguments:
            - mean (torch.Tensor)   : mean of diagional gaussian
            - logvar (torch.Tensor) : log-variance of diagonal gaussian
        '''
        # Generate noise from standard gaussian
        eps = torch.randn(mean.shape, requires_grad=False, dtype=torch.float32).to(torch.get_default_dtype()).to(self.device)
        # Scale and shift by mean and standard deviation
        return torch.exp(logvar*0.5)*eps + mean
    
    def initialize_hidden_states(self, input):
        '''
        initialize_hidden_states()
        
        Initialize hidden states of recurrent networks
        '''
        
        # Check dimensions
        self.steps_size, self.batch_size, input_size = input.shape
        assert input_size == self.input_size, 'Input is expected to have dimensions [%i, %i, %i]'%(self.steps_size, self.batch_size, self.input_size)
        
        g_encoder_state  = (torch.ones(self.batch_size, 2,  self.g_encoder_size, device=self.device) * self.g_encoder_init).permute(1, 0, 2)
        if self.c_encoder_size > 0 and self.controller_size > 0 and self.u_latent_size > 0:
            c_encoder_state  = (torch.ones(self.batch_size, 2,  self.c_encoder_size, device=self.device) * self.c_encoder_init).permute(1, 0, 2)
            controller_state = torch.ones(self.batch_size, self.controller_size, device=self.device) * self.controller_init
            return g_encoder_state, c_encoder_state, controller_state
        else:
            return g_encoder_state, None, None
    
    def _gp_to_normal(self, gp_mean, gp_logvar, gp_logtau, process):
        '''
        _gp_to_normal(gp_mean, gp_logvar, gp_logtau, process)
        
        Convert gaussian process with given process mean, process log-variance, process tau, and realized process
        to mean and log-variance of diagonal Gaussian for each time-step
        '''
        
        mean   = gp_mean * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        logvar = gp_logvar * torch.ones(1, process.shape[1], process.shape[2], device=self.device)
        
        mean   = torch.cat((mean, gp_mean + (process[:-1] - gp_mean) * torch.exp(-1/gp_logtau.exp())))
        logvar = torch.cat((logvar, torch.log(1 - torch.exp(-1/gp_logtau.exp()).pow(2)) + gp_logvar * torch.ones(process.shape[0]-1, process.shape[1], process.shape[2], device=self.device)))
        return mean.permute(1, 0, 2), logvar.permute(1, 0, 2)
    
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
                self.normalize_factors()
                
    def normalize_factors(self):
        self.generator.fc_factors.weight.data = F.normalize(self.generator.fc_factors.weight.data, dim=1)
        
    def change_parameter_grad_status(self, step, optimizer, scheduler, loading_checkpoint=False):
        return optimizer, scheduler
    
    def kl_div(self):
        kl = kldiv_gaussian_gaussian(post_mu  = self.g_posterior_mean,
                                     post_lv  = self.g_posterior_logvar,
                                     prior_mu = self.g_prior_mean,
                                     prior_lv = self.g_prior_logvar)
        if self.u_latent_size > 0:
            kl += kldiv_gaussian_gaussian(post_mu  = self.u_posterior_mean,
                                          post_lv  = self.u_posterior_logvar,
                                          prior_mu = self.u_prior_mean,
                                          prior_lv = self.u_prior_logvar)
        return kl
    
class LFADS_SingleSession_Net(LFADS_Net):
    
    def __init__(self, input_size, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu', output_nonlin='exp'):
        
        super(LFADS_SingleSession_Net, self).__init__(input_size = input_size, factor_size = factor_size, prior = prior,
                                                      g_encoder_size   = g_encoder_size, c_encoder_size = c_encoder_size,
                                                      g_latent_size    = g_latent_size, u_latent_size = u_latent_size,
                                                      controller_size  = controller_size, generator_size = generator_size,
                                                      clip_val=clip_val, dropout=dropout, max_norm = max_norm, deep_freeze = deep_freeze,
                                                      do_normalize_factors=do_normalize_factors, factor_bias = factor_bias, device=device)
        
        self.fc_logrates = nn.Linear(in_features= self.factor_size, out_features= self.input_size)
        print(output_nonlin)
        self.output_nonlin = output_nonlin
        self.nonlin = nn.Softplus()
        self.initialize_weights()
        
    def forward(self, input):
        factors, gen_inputs = super(LFADS_SingleSession_Net, self).forward(input.permute(1, 0, 2))
        if self.output_nonlin == 'exp':
            recon = {'rates' : self.fc_logrates(factors).exp()}
        elif self.output_nonlin == 'softplus':
            recon = {'rates' : self.nonlin(self.fc_logrates(factors))}

        recon['data'] = recon['rates'].clone().permute(1, 0, 2)
        return recon, (factors, gen_inputs)
    
class LFADS_MultiSession_Net(LFADS_Net):
    
    def __init__(self, W_in_list, W_out_list, b_in_list, b_out_list, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        
        super(LFADS_MultiSession_Net, self).__init__(input_size = factor_size, factor_size = factor_size, prior = prior,
                                                     g_encoder_size   = g_encoder_size, c_encoder_size = c_encoder_size,
                                                     g_latent_size    = g_latent_size, u_latent_size = u_latent_size,
                                                     controller_size  = controller_size, generator_size = generator_size,
                                                     clip_val=clip_val, dropout=dropout, max_norm = max_norm, deep_freeze = deep_freeze,
                                                     do_normalize_factors=do_normalize_factors, factor_bias = factor_bias, device=device)
        
        for idx, (W_in, b_in, W_out, b_out) in enumerate(zip(W_in_list, b_in_list, W_out_list, b_out_list)):
            assert W_in.shape[1] == self.factor_size, 'Read in matrix should have dim 1 = %i, but has dims [%i, %i]'%(self.factor_size, W_in.shape[0], W_in.shape[1])
            assert W_out.shape[0] == self.factor_size, 'Read out matrix should have dim 0 = %i, but has dims [%i, %i]'%(self.factor_size, W_out.shape[0], W_out.shape[1])
            setattr(self, 'fc_input_%i'%idx, nn.Linear(in_features=W_in.shape[0], out_features=self.factor_size))
            setattr(self, 'fc_logrates_%i'%idx, nn.Linear(in_features=self.factor_size, out_features=W_in.shape[0]))
            
            getattr(self, 'fc_input_%i'%idx).weight.data = W_in.permute(1, 0)
            getattr(self, 'fc_input_%i'%idx).bias.data = b_in
            getattr(self, 'fc_logrates_%i'%idx).weight.data = W_out.permute(1, 0)
            getattr(self, 'fc_logrates_%i'%idx).bias.data = b_out
            
#             getattr(self, 'fc_input_%i'%idx).weight.requires_grad = False
#             getattr(self, 'fc_logrates_%i'%idx).weight.requires_grad = False
#             getattr(self, 'fc_input_%i'%idx).bias.requires_grad = False
#             getattr(self, 'fc_logrates_%i'%idx).bias.requires_grad = False
            
    def forward(self, input):
        aligned_input = getattr(self, 'fc_input_%i'%input.session)(input).permute(1, 0, 2)
        factors, gen_inputs = super(LFADS_MultiSession_Net, self).forward(aligned_input)
        recon = {'rates' : getattr(self, 'fc_logrates_%i'%input.session)(factors).exp().permute(1, 0, 2)}
        recon['data'] = recon['rates'].clone()
        return recon, (factors, gen_inputs)
        
    
class LFADS_Encoder(nn.Module):
    '''
    LFADS_Encoder
    
    LFADS Encoder Network 
    
    __init__(self, input_size, g_encoder_size, g_latent_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0)
    
    Required Arguments:
        - input_size (int):  size of input dimensions
        - g_encoder_size (int):  size of generator encoder network
        - g_latent_size (int): size of generator ic latent variable
        
    Optional Arguments:
        - c_encoder_size (int): size of controller encoder network
        - dropout (float): dropout probability
        - clip_val (float): RNN hidden state value limit
        
    '''
    def __init__(self, input_size, g_encoder_size, g_latent_size, c_encoder_size= 0, dropout= 0.0, clip_val= 5.0):
        super(LFADS_Encoder, self).__init__()
        self.input_size      = input_size
        self.g_encoder_size  = g_encoder_size
        self.c_encoder_size  = c_encoder_size
        self.g_latent_size   = g_latent_size
        self.clip_val        = clip_val

        self.dropout = nn.Dropout(dropout)
        
        # g Encoder BiRNN
        self.gru_g_encoder  = nn.GRU(input_size=self.input_size, hidden_size=self.g_encoder_size, bidirectional=True)
        # g Linear mapping
        self.fc_g0_theta    = nn.Linear(in_features= 2 * self.g_encoder_size, out_features= self.g_latent_size * 2)
        
        if self.c_encoder_size > 0:
            # c encoder BiRNN
            self.gru_c_encoder  = nn.GRU(input_size=self.input_size, hidden_size=self.c_encoder_size, bidirectional=True)
            
    def forward(self, input, hidden):
        self.gru_g_encoder.flatten_parameters()
        if self.c_encoder_size > 0:
            self.gru_c_encoder.flatten_parameters()
        gru_g_encoder_init, gru_c_encoder_init = hidden
        
        # Run bidirectional RNN over data
        out_gru_g_enc, hidden_gru_g_enc = self.gru_g_encoder(self.dropout(input), gru_g_encoder_init.contiguous())
        hidden_gru_g_enc = self.dropout(hidden_gru_g_enc.clamp(min=-self.clip_val, max=self.clip_val))
        hidden_gru_g_enc = torch.cat((hidden_gru_g_enc[0], hidden_gru_g_enc[1]), dim=1)
        
        g0_mean, g0_logvar = torch.split(self.fc_g0_theta(hidden_gru_g_enc), self.g_latent_size, dim=1)
        
        if self.c_encoder_size > 0:
            out_gru_c_enc, hidden_gru_c_enc = self.gru_c_encoder(self.dropout(input), gru_c_encoder_init.contiguous())
            out_gru_c_enc = out_gru_c_enc.clamp(min=-self.clip_val, max=self.clip_val)
        
            return g0_mean, g0_logvar, out_gru_g_enc, out_gru_c_enc
        
        else:
            
            return g0_mean, g0_logvar, out_gru_g_enc, None
        
class LFADS_ControllerCell(nn.Module):
    
    def __init__(self, input_size, controller_size, u_latent_size, dropout = 0.0, clip_val=5.0):
        super(LFADS_ControllerCell, self).__init__()
        self.input_size      = input_size
        self.controller_size = controller_size
        self.u_latent_size   = u_latent_size
        self.clip_val        = clip_val
        
        self.dropout = nn.Dropout(dropout)
        
        self.gru_controller  = LFADS_GenGRUCell(input_size  = self.input_size, hidden_size = self.controller_size)
        self.fc_u_theta = nn.Linear(in_features = self.controller_size, out_features=self.u_latent_size * 2)
        
    def forward(self, input, hidden):
        controller_state = hidden
        controller_state = self.gru_controller(self.dropout(input), controller_state)
        controller_state = controller_state.clamp(-self.clip_val, self.clip_val)
        u_mean, u_logvar = torch.split(self.fc_u_theta(controller_state), self.u_latent_size, dim=1)
        return u_mean, u_logvar, controller_state
    
class LFADS_GeneratorCell(nn.Module):
    
    def __init__(self, input_size, generator_size, factor_size, dropout = 0.0, clip_val = 5.0, factor_bias = False):
        super(LFADS_GeneratorCell, self).__init__()
        self.input_size = input_size
        self.generator_size = generator_size
        self.factor_size = factor_size
        
        self.dropout = nn.Dropout(dropout)
        self.clip_val = clip_val
        
        self.gru_generator = LFADS_GenGRUCell(input_size=input_size, hidden_size=generator_size)
        self.fc_factors = nn.Linear(in_features=generator_size, out_features=factor_size, bias=factor_bias)
        
    def forward(self, input, hidden):
        
        generator_state = hidden
        generator_state = self.gru_generator(input, generator_state)
        generator_state = generator_state.clamp(min=-self.clip_val, max=self.clip_val)
        factor_state    = self.fc_factors(self.dropout(generator_state))
        
        return generator_state, factor_state

    
class Identity(nn.Module):
    def __init__(self, in_features, out_features):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

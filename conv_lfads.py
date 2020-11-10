import torch
import torch.nn as nn
from lfads import LFADS_Net
from svlae import SVLAE_Net
import time
import pdb

class Conv3d_LFADS_Net(nn.Module):
    def __init__(self,
                input_dims = (100,128,128), conv_type = '2d', #(100, 128, 128)
                channel_dims = (16, 32), 
                obs_encoder_size = 32, obs_latent_size = 64,
                obs_controller_size = 32, 
                conv_dense_size = 64, factor_size = 4,
                g_encoder_size  = 64, c_encoder_size = 64,
                g_latent_size   = 64, u_latent_size  = 1,
                controller_size = 64, generator_size = 64,
                prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                obs_params = {'gain' : {'value' : 1.0, 'learnable' : False},
                               'bias' : {'value' : 0.0, 'learnable' : False},
                               'tau'  : {'value' : 10., 'learnable' : False},
                               'var'  : {'value' : 0.1, 'learnable' : True}},
                deep_unfreeze_step = 1600, obs_early_stop_step = 2000, 
                generator_burn = 0, obs_continue_step  = 8000, 
                ar1_start_step = 4000, clip_val=5.0, 
                max_norm = 200, lfads_dropout=0.0, 
                conv_dropout=0.0,do_normalize_factors=True, 
                factor_bias = False, device='cpu'):
        super(Conv3d_LFADS_Net, self).__init__()
        
        self.conv_type = conv_type
        self.factor_size = factor_size
        self.obs_encoder_size = obs_encoder_size
        self.obs_latent_size = obs_latent_size
        self.obs_controller_size = obs_controller_size
        self.g_encoder_size = g_encoder_size
        self.c_encoder_size = c_encoder_size
        self.g_latent_size = g_latent_size
        self.u_latent_size = u_latent_size
        self.controller_size = controller_size
        self.generator_size = generator_size
        self.clip_val = clip_val
        self.max_norm = max_norm
        self.do_normalize_factors = do_normalize_factors
        self.factor_bias = factor_bias
        
        self.device= device
        self.input_dims = input_dims
        self.channel_dims = (1,) + channel_dims
        self.conv_layers = nn.ModuleList()
        self.conv_dense_size = conv_dense_size
        
        if self.conv_type == '2d':
            layer_dims = self.input_dims[1:]
            for n in range(1, len(self.channel_dims)):
                self.conv_layers.add_module('{}{}'.format('block', n),
                                            Conv2d_Block_1step(input_dims = layer_dims,
                                                               in_f = self.channel_dims[n-1],
                                                               out_f= self.channel_dims[n]))
                layer_dims = getattr(self.conv_layers, '{}{}'.format('block', n)).get_output_dims()
        
            self.deconv_layers = nn.ModuleList()
            for n in reversed(range(1, len(self.channel_dims))):
                self.deconv_layers.add_module('{}{}'.format('block', n),
                                              ConvTranspose2d_Block_1step(in_f = self.channel_dims[n],
                                                                          out_f= self.channel_dims[n-1]))
            
        elif self.conv_type == '3d':
            layer_dims = self.input_dims
            for n in range(1, len(self.channel_dims)):
                self.conv_layers.add_module('{}{}'.format('block', n),
                                            Conv3d_Block_1step(input_dims = layer_dims,
                                                               in_f = self.channel_dims[n-1],
                                                               out_f= self.channel_dims[n]))
                layer_dims = getattr(self.conv_layers, '{}{}'.format('block', n)).get_output_dims()

            self.deconv_layers = nn.ModuleList()
            for n in reversed(range(1, len(self.channel_dims))):
                self.deconv_layers.add_module('{}{}'.format('block', n),
                                              ConvTranspose3d_Block_1step(in_f = self.channel_dims[n],
                                                                          out_f= self.channel_dims[n-1]))

        elif self.conv_type == 'fix':
            pass
        
        
            
            
        # Placeholder
        if self.conv_type == '3d':
            self.conv_output_size = int(torch._np.prod(layer_dims[1:]) * self.channel_dims[-1])
        else:
            self.conv_output_size = int(torch._np.prod(layer_dims[0:]) * self.channel_dims[-1])
            
        self.conv_dense_size = self.conv_dense_size
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.conv_dense_1 = nn.Linear(in_features= self.conv_output_size,
                                      out_features= self.conv_dense_size)
        self.conv_dense_2 = nn.Linear(in_features= self.factor_size,
                                      out_features = self.conv_dense_size)
        self.conv_dense_3 = nn.Linear(in_features= self.conv_dense_size,
                                      out_features = self.conv_output_size)
        self.RELU = nn.ReLU()
        
        
#         self.lfads_param = dict()
        print(self.device)
        print(torch.cuda.device_count())
        # self.lfads = LFADS_Net(input_size= self.conv_dense_size,
        #                     g_encoder_size=self.g_encoder_size,
        #                     c_encoder_size=self.c_encoder_size,
        #                     g_latent_size=self.g_latent_size,
        #                     u_latent_size=self.u_latent_size,
        #                     controller_size=self.controller_size,
        #                     generator_size=self.generator_size,
        #                     factor_size=self.factor_size,
        #                     prior=prior,
        #                     clip_val=self.clip_val,
        #                     dropout=lfads_dropout,
        #                     max_norm=self.max_norm,
        #                     do_normalize_factors=self.do_normalize_factors,
        #                     factor_bias=self.factor_bias,
        #                     device= self.device)
        self.calfads = SVLAE_Net(input_size = self.conv_dense_size,
                    factor_size           = self.factor_size,
                    obs_encoder_size      = self.obs_encoder_size,
                    obs_latent_size       = self.obs_latent_size,
                    obs_controller_size   = self.obs_controller_size,
                    deep_g_encoder_size   = self.g_encoder_size,
                    deep_c_encoder_size   = self.c_encoder_size,
                    deep_g_latent_size    = self.g_latent_size,
                    deep_u_latent_size    = self.u_latent_size,
                    deep_controller_size  = self.controller_size,
                    generator_size        = self.generator_size,
                    prior                 = prior,
                    clip_val              = self.clip_val,
                    generator_burn        = generator_burn,
                    dropout               = lfads_dropout,
                    do_normalize_factors  = self.do_normalize_factors,
                    factor_bias           = self.factor_bias,
                    max_norm              = self.max_norm,
                    deep_unfreeze_step    = deep_unfreeze_step,
                    obs_early_stop_step   = obs_early_stop_step,
                    obs_continue_step     = obs_continue_step,
                    ar1_start_step        = ar1_start_step,
                    obs_params            = obs_params,
                    device                = self.device)
        
        self.register_parameter('u_posterior_mean',None)
        self.register_parameter('u_posterior_logvar',None)
        self.register_parameter('g_posterior_mean',None)
        self.register_parameter('g_posterior_logvar',None)

        self.register_parameter('g_prior_mean',self.calfads.deep_model.g_prior_mean)
        self.register_buffer('g_prior_logvar',self.calfads.deep_model.g_prior_logvar)
        self.register_parameter('u_prior_mean',self.calfads.obs_model.u_prior_mean)
        self.register_buffer('u_prior_logvar',self.calfads.obs_model.u_prior_logvar)
        
        
        
    def forward(self, x):
        
        
        batch_size, num_ch, seq_len, w, h = x.shape
        if self.conv_type == '3d':
            
            frame_per_block = 5
            num_blocks = int(seq_len/frame_per_block)

            x = x.view(batch_size, num_ch, num_blocks, frame_per_block, w, h).contiguous()
            x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
            x = x.view(batch_size * num_blocks, num_ch, frame_per_block, w, h).contiguous()
        
        else:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(batch_size*seq_len,num_ch,w,h).contiguous()
            

        Ind = list()
        conv_tic = time.time()
        for n, layer in enumerate(self.conv_layers):
            x, ind1 = layer(x)
            Ind.append(ind1)
        conv_toc = time.time()

        num_out_ch = x.shape[1]
        w_out = x.shape[-1]
        h_out = x.shape[-2]
        
        if self.conv_type == '3d':
            x = x.view(batch_size, num_blocks, num_out_ch, frame_per_block, w_out, h_out).contiguous()
            x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
            x = x.view(batch_size, num_out_ch, seq_len, w_out, h_out).contiguous()
        
        else:
            x = x.view(batch_size, seq_len, num_out_ch, w_out, h_out).contiguous()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        

        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        
        x = self.conv_dense_1(x.view(batch_size, seq_len, w_out * h_out * num_out_ch))
        x = self.RELU(x)
        conv_out = x
        
        # x = x.permute(1, 0, 2)
        lfads_tic = time.time()
        # factors, gen_inputs = self.lfads(x)
        
        recon_calfads, (factors, deep_gen_inputs) = self.calfads(x)
        lfads_toc = time.time()
        # print('conv t: ',conv_toc - conv_tic,' lfads t: ',lfads_toc - lfads_tic)
        # x = factors
        x = recon_calfads['data']
        # x = x.permute(1, 0, 2)
        # x = self.conv_dense_2(x).exp()
        deconv_in = x
        x = self.conv_dense_3(x)
        # x = self.RELU(x)
        
        # call LFADS here:
        # x should be reshaped for LFADS [time x batch x cells]:
        # 
        # LFADS output should be also reshaped back for the conv decoder
        
        x = x.reshape(x.shape[0], x.shape[1], num_out_ch, w_out, h_out)
        x = x.permute(0, 2, 1, 3, 4)
        
        if self.conv_type == '3d':
            x = x.view(batch_size, num_out_ch, num_blocks, frame_per_block, w_out, h_out).contiguous()
            x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
            x = x.view(batch_size * num_blocks, num_out_ch, frame_per_block, w_out, h_out).contiguous()
        else:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(batch_size * seq_len, num_out_ch, w_out, h_out).contiguous()

        for layer, ind in list(zip(self.deconv_layers, reversed(Ind))):
            x = layer(x, ind)

        if self.conv_type == '3d':
            x = x.view(batch_size, num_blocks, 1, frame_per_block, w, h).contiguous()
            x = x.permute(0, 2, 1, 3, 4, 5)
            x = x.view(batch_size, 1, seq_len, w, h)
        else:
            x = x.view(batch_size, seq_len, 1, w, h).contiguous()
            x = x.permute((0, 2, 1, 3, 4)).contiguous()
            
        # g_posterior = dict()
        # g_posterior_mean = self.lfads.g_posterior_mean
        # g_posterior_logvar = self.lfads.g_posterior_logvar
        u_posterior_mean = self.calfads.obs_model.u_posterior_mean
        u_posterior_logvar = self.calfads.obs_model.u_posterior_logvar

        g_posterior_mean = self.calfads.deep_model.g_posterior_mean
        g_posterior_logvar = self.calfads.deep_model.g_posterior_logvar
        
        recon = {}
        recon['data'] = x
        recon['spikes'] = recon_calfads['spikes']
        recon['rates'] = recon_calfads['rates']

        return recon, (factors, deep_gen_inputs), (g_posterior_mean,g_posterior_logvar), (u_posterior_mean, u_posterior_logvar), conv_out
    
    def normalize_factors(self):
        self.calfads.deep_model.normalize_factors()
        
    def change_parameter_grad_status(self, step, optimizer, scheduler, loading_checkpoint=False):
        return optimizer, scheduler

class _ConvNd_Block(nn.ModuleList):
    def __init__(self, input_dims):
        super(_ConvNd_Block, self).__init__()
        
        self.input_dims = input_dims
        
    def forward(self, x):
        ind = None
        for layer in self:
            if nn.modules.pooling._MaxPoolNd in type(layer).__bases__ and layer.return_indices:
                x, ind = layer(x)
            else:
                x = layer(x)
        return x, ind
    
    def get_output_dims(self):
        def layer_out_dim(in_dim, layer):
            padding = layer.padding
            kernel_size = layer.kernel_size
            dilation = layer.dilation
            stride = layer.stride
            
            def out_dim(in_dim, padding, dilation, kernel_dim, stride):
                return int((in_dim + 2 * padding - dilation * (kernel_dim - 1) - 1)/stride + 1)
            
            return tuple([out_dim(i,p,d,k,s) for i,p,d,k,s in zip(in_dim,
                                                                  padding,
                                                                  dilation,
                                                                  kernel_size,
                                                                  stride)])
        
        dims = self.input_dims
        for m in self:
            parents = type(m).__bases__
            if nn.modules.conv._ConvNd in parents or nn.modules.pooling._MaxPoolNd in parents:
                dims = layer_out_dim(dims, m)        
                
        
        return dims
    
class Conv3d_Block_2step(_ConvNd_Block):
    def __init__(self, in_f, out_f,
                 kernel_size=(3, 3, 3),
                 dilation=(1, 1, 1),
                 padding=(1, 1, 1),
                 stride=(1, 1, 1),
                 pool_size=(1, 4, 4),
                 input_dims=(100, 100, 100)):
        super(Conv3d_Block_2step, self).__init__(input_dims)
        
        self.add_module('conv1', nn.Conv3d(in_f, out_f,
                                           kernel_size= kernel_size, 
                                           padding= padding,
                                           dilation = dilation,
                                           stride= stride))        
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', nn.Conv3d(out_f, out_f, 
                                           kernel_size= kernel_size, 
                                           padding= padding,
                                           dilation= dilation,
                                           stride = stride))
        self.add_module('pool1', nn.MaxPool3d(kernel_size= pool_size,
                                              stride= pool_size,
                                              padding=(0, 0, 0),
                                              dilation=(1, 1, 1),
                                              return_indices= True))
        self.add_module('relu2', nn.ReLU())
               
        self.output_dims = self.get_output_dims()
        
        
class Conv2d_fix(nn.Conv2d):
    def __init__(self, in_f, out_f,
                 kernel_size = 11,
                 stride=1, 
                 padding=5,
                 radius=4,
                 center = (5,5)):
        super(Conv2d_fix, self).__init__(in_f, out_f, kernel_size = kernel_size, stride = stride, padding = kernel_size//2)
        
        self.center = center
        self.radius = radius
        self.kernel_size = kernel_size
    
        
        self.Weights = nn.Parameter(self.make_weights())
        
    def forward(self,x):
        return super(Conv2d_fix)._conv_forward(x, self.Weights)
        

    
    def make_weights(self):
        
        import skimage.draw as draw
        rr, cc = draw.circle(r = self.center[0], c = self.center[1] , radius=self.radius)
        
        w = torch.zeros((1,1,self.kernel_size,self.kernel_size))
        w[0,0,rr,cc] += 1
        
        return w
        
        
        
class Conv2d_Block_1step(_ConvNd_Block):
    
    def __init__(self, in_f, out_f,
                 kernel_size=(3, 3),
                 dilation=(1, 1),
                 padding=(1, 1),
                 stride=(1, 1),
                 pool_size=(2, 2),
                 input_dims=(128, 128)):
        super(Conv2d_Block_1step, self).__init__(input_dims)
        
        self.add_module('conv1', nn.Conv2d(in_f, out_f,
                                           kernel_size= kernel_size, 
                                           padding= padding,
                                           dilation = dilation,
                                           stride= stride))
        self.add_module('relu1', nn.ReLU())
        self.add_module('pool1', nn.MaxPool2d(kernel_size= pool_size,
                                              stride= pool_size,
                                              padding=(0, 0),
                                              dilation=(1, 1),
                                              return_indices= True))
        
    
    
class Conv3d_Block_1step(_ConvNd_Block):

    def __init__(self, in_f, out_f,
                 kernel_size=(1, 3, 3),
                 dilation=(1, 1, 1),
                 padding=(0, 1, 1),
                 stride=(1, 1, 1),
                 pool_size=(1, 2, 2),
                 input_dims=(100, 100, 100)):
        super(Conv3d_Block_1step, self).__init__(input_dims)
    
        
        self.add_module('conv1', nn.Conv3d(in_f, out_f,
                                           kernel_size= kernel_size, 
                                           padding= padding,
                                           dilation = dilation,
                                           stride= stride))
        self.add_module('relu1', nn.ReLU())
        self.add_module('pool1', nn.MaxPool3d(kernel_size= pool_size,
                                              stride= pool_size,
                                              padding=(0, 0, 0),
                                              dilation=(1, 1, 1),
                                              return_indices= True))
        
    
class _ConvTransposeNd_Block(nn.ModuleList):
    def __init__(self):
        super(_ConvTransposeNd_Block, self).__init__()
    
    def forward(self, x, ind):
        for layer in self:
            if nn.modules.pooling._MaxUnpoolNd in type(layer).__bases__:
                x = layer(x, ind)
            else:
                x = layer(x)
        return x
    
class ConvTranspose2d_Block_1step(_ConvTransposeNd_Block):
    def __init__(self, in_f, out_f):
        super(ConvTranspose2d_Block_1step, self).__init__()
        
        self.add_module('unpool1', nn.MaxUnpool2d(kernel_size=(2,2)))
        self.add_module('deconv1', nn.ConvTranspose2d(in_channels= in_f,
                                          out_channels= out_f,
                                          kernel_size= (3,3),
                                          padding= (1,1), 
                                          dilation= (1,1)))
    
class ConvTranspose3d_Block_1step(_ConvTransposeNd_Block):
    def __init__(self, in_f, out_f):
        super(ConvTranspose3d_Block_1step, self).__init__()
        
        self.add_module('unpool1', nn.MaxUnpool3d(kernel_size=(1,2,2)))
        self.add_module('deconv1', nn.ConvTranspose3d(in_channels= in_f,
                                          out_channels= out_f,
                                          kernel_size= (1,3,3),
                                          padding= (0,1,1), 
                                          dilation= (1,1,1)))

        self.add_module('relu1', nn.ReLU())


if __name__ == "__main__":
    
    import pdb
    
    from utils import load_parameters
    
    x = torch.rand((10, 1, 20, 128, 128)).to('cuda')
    print('input size', x.shape)
    batch_size, num_ch, seq_len, w, h = x.shape
    
    hyperparams = load_parameters('./hyperparameters/lorenz/conv3d_lfads.yaml')
    
    model = Conv3d_LFADS_Net(input_dims             = (100,128,128), 
                             conv_type = '2d',
                             channel_dims           = hyperparams['model']['channel_dims'], 
                             obs_encoder_size       = hyperparams['model']['obs_encoder_size'], 
                             obs_latent_size        = hyperparams['model']['obs_latent_size'],
                             obs_controller_size    = hyperparams['model']['obs_controller_size'], 
                             conv_dense_size        = hyperparams['model']['conv_dense_size'], 
                             factor_size            = hyperparams['model']['factor_size'],
                             g_encoder_size         = hyperparams['model']['g_encoder_size'], 
                             c_encoder_size         = hyperparams['model']['c_encoder_size'],
                             g_latent_size          = hyperparams['model']['g_latent_size'], 
                             u_latent_size          = hyperparams['model']['u_latent_size'],
                             controller_size        = hyperparams['model']['controller_size'], 
                             generator_size         = hyperparams['model']['generator_size'],
                             prior                  = hyperparams['model']['prior'],
                             obs_params             = hyperparams['model']['obs'],
                             deep_unfreeze_step     = hyperparams['model']['deep_unfreeze_step'], 
                             obs_early_stop_step    = hyperparams['model']['obs_early_stop_step'], 
                             generator_burn         = hyperparams['model']['generator_burn'], 
                             obs_continue_step      = hyperparams['model']['obs_continue_step'], 
                             ar1_start_step         = hyperparams['model']['ar1_start_step'], 
                             clip_val               = hyperparams['model']['clip_val'], 
                             max_norm               = hyperparams['model']['max_norm'], 
                             lfads_dropout          = hyperparams['model']['lfads_dropout'], 
                             conv_dropout           = hyperparams['model']['conv_dropout'],
                             do_normalize_factors   = hyperparams['model']['normalize_factors'], 
                             factor_bias            = hyperparams['model']['factor_bias'], 
                             device                 = 'cuda').to('cuda')
    
    
    
    recon, (factors, deep_gen_inputs), (g_posterior_mean,g_posterior_logvar), (u_posterior_mean, u_posterior_logvar), conv_out = model(x)
    pdb.set_trace()
#     from synthetic_data import *
    


#     lorenz = LorenzSystem(num_inits= 100,
#                                 dt= 0.01)

#     net = EmbeddedLowDNetwork(low_d_system = lorenz,
#                                 net_size = 64,
#                                 base_rate = 1.0,
#                                 dt = 0.01)

#     Ca_synth = SyntheticCalciumDataGenerator(net, 100, trainp = 0.8,
#                      burn_steps = 1000, num_trials = 10, num_steps= 20,
#                      tau_cal=0.1, dt_cal= 0.01, sigma=0.2,
#                      frame_width=128, frame_height=128, cell_radius=4, save=True)
#     data_dict = Ca_synth.generate_dataset()
    
#     train_dl    = torch.utils.data.DataLoader(SyntheticCalciumVideoDataset(traces= data_dict['train_fluor'], cells=data_dict['cells'], device='cuda'), batch_size=1)
    
#     conv = Conv2d_fix(in_f = 1, out_f = 1, kernel_size = 11, stride=1, padding=5, radius=4, center = (5,5))
#     out_data = torch.zeros((800,1,1,20,128,128))
#     in_data = torch.zeros((800,1,1,20,128,128)).to('cuda')
    
#     n=0
#     for data in train_dl:
#     #     print(data[0].shape)
#     #     in_data[n,:] = data[0]
#     #     out_data[n,:] = torch.nn.functional.conv2d(in_data[n,:],W,stride=1, padding=5)#conv.conv3d_forward(in_data,W)
#         for t in range(0,data[0].shape[2]):
#             in_data[n,:,:,t,:,:] = data[0][:,:,t,:,:]
#             out_data[n,:,:,t,:,:] = conv(in_data[n,:,:,t,:,:])#conv.conv3d_forward(in_data,W)
#             pdb.set_trace()
#         n+=1

import torch
import torch.nn as nn
from lfads import LFADS_Net
import time
import pdb

class Conv3d_LFADS_Net(nn.Module):
    def __init__(self, input_dims = (100, 128, 128), channel_dims = (16, 32), 
                 conv_dense_size = 64, factor_size = 4,
                 g_encoder_size  = 64, c_encoder_size = 64,
                 g_latent_size   = 64, u_latent_size  = 1,
                 controller_size = 64, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, max_norm = 200, lfads_dropout=0.0, conv_dropout=0.0,
                 do_normalize_factors=True, factor_bias = False, device='cpu'):
        super(Conv3d_LFADS_Net, self).__init__()
        
        self.factor_size = factor_size
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
            
            
        # Placeholder
        self.conv_output_size = int(torch._np.prod(layer_dims[1:]) * self.channel_dims[-1])
        self.conv_dense_size = self.conv_dense_size
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.conv_dense_1 = nn.Linear(in_features= self.conv_output_size,
                                      out_features= self.conv_dense_size)
        self.conv_dense_2 = nn.Linear(in_features= self.factor_size,
                                      out_features = self.conv_output_size)
        
        
#         self.lfads_param = dict()
        print(self.device)
        print(torch.cuda.device_count())
        self.lfads = LFADS_Net(input_size= self.conv_dense_size,
                               g_encoder_size=self.g_encoder_size,
                               c_encoder_size=self.c_encoder_size,
                               g_latent_size=self.g_latent_size,
                               u_latent_size=self.u_latent_size,
                               controller_size=self.controller_size,
                               generator_size=self.generator_size,
                               factor_size=self.factor_size,
                               prior=prior,
                               clip_val=self.clip_val,
                               dropout=lfads_dropout,
                               max_norm=self.max_norm,
                               do_normalize_factors=self.do_normalize_factors,
                               factor_bias=self.factor_bias,
                               device= self.device)
        
        self.register_buffer('g_posterior_mean',None)
        self.register_buffer('g_posterior_logvar',None)
        self.register_buffer('g_prior_mean',self.lfads.g_prior_mean)
        self.register_buffer('g_prior_logvar',self.lfads.g_prior_logvar)
        
#         self.lfads_param['g_posterior_mean'] = self.lfads.g_posterior_mean
#         self.lfads_param['g_posterior_logvar'] = self.lfads.g_posterior_logvar
#         self.lfads_param['g_prior_mean'] = self.lfads.g_prior_mean
#         self.lfads_param['g_prior_logvar'] = self.lfads.g_prior_logvar
        
        
        
    def forward(self, x):
        
        frame_per_block = 10
        batch_size, num_ch, seq_len, w, h = x.shape
        num_blocks = int(seq_len/frame_per_block)
        
        x = x.view(batch_size, num_ch, num_blocks, frame_per_block, w, h).contiguous()
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        x = x.view(batch_size * num_blocks, num_ch, frame_per_block, w, h).contiguous()
        

        Ind = list()
        conv_tic = time.time()
        for n, layer in enumerate(self.conv_layers):
            x, ind1 = layer(x)
            Ind.append(ind1)
        conv_toc = time.time()

        num_out_ch = x.shape[1]
        w_out = x.shape[3]
        h_out = x.shape[4]
        x = x.view(batch_size, num_blocks, num_out_ch, frame_per_block, w_out, h_out).contiguous()
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        
        x = x.view(batch_size, num_out_ch, seq_len, w_out, h_out).contiguous()

        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0],x.shape[1],-1)
#         pdb.set_trace()
        x = self.conv_dense_1(x.view(batch_size, seq_len, w_out * h_out * num_out_ch))
        
        x = x.permute(1, 0, 2)
        lfads_tic = time.time()
        factors, gen_inputs = self.lfads(x)
        lfads_toc = time.time()
        # print('conv t: ',conv_toc - conv_tic,' lfads t: ',lfads_toc - lfads_tic)
        x = factors
        x = x.permute(1, 0, 2)
        x = self.conv_dense_2(x)
        
        # call LFADS here:
        # x should be reshaped for LFADS [time x batch x cells]:
        # 
        # LFADS output should be also reshaped back for the conv decoder
        
        x = x.reshape(x.shape[0], x.shape[1], num_out_ch, w_out, h_out)
        x = x.permute(0, 2, 1, 3, 4)
        
        x = x.view(batch_size, num_out_ch, num_blocks, frame_per_block, w_out, h_out).contiguous()
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        x = x.view(batch_size * num_blocks, num_out_ch, frame_per_block, w_out, h_out).contiguous()

        for layer, ind in list(zip(self.deconv_layers, reversed(Ind))):
            x = layer(x, ind)
        
        x = x.view(batch_size, num_blocks, 1, frame_per_block, w, h).contiguous()
        x = x.permute(0, 2, 1, 3, 4, 5)
        x = x.view(batch_size, 1, seq_len, w, h)
        
        g_posterior = dict()
        g_posterior['mean'] = self.lfads.g_posterior_mean
        g_posterior['logvar'] = self.lfads.g_posterior_logvar
        
        recon = {'data' : x}

        return recon, (factors, gen_inputs), g_posterior
    
    def normalize_factors(self):
        self.lfads.normalize_factors()
        
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
        
class Conv3d_Block_1step(_ConvNd_Block):
    def __init__(self, in_f, out_f,
                 kernel_size=(3, 3, 3),
                 dilation=(1, 1, 1),
                 padding=(1, 1, 1),
                 stride=(1, 1, 1),
                 pool_size=(1, 4, 4),
                 input_dims=(100, 100, 100)):
        super(Conv3d_Block_1step, self).__init__(input_dims)
        
        self.add_module('conv1', nn.Conv3d(in_f, out_f,
                                           kernel_size= kernel_size, 
                                           padding= padding,
                                           dilation = dilation,
                                           stride= stride))        
        self.add_module('pool1', nn.MaxPool3d(kernel_size= pool_size,
                                              stride= pool_size,
                                              padding=(0, 0, 0),
                                              dilation=(1, 1, 1),
                                              return_indices= True))
        self.add_module('relu1', nn.ReLU())
    
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
    
class ConvTranspose3d_Block_1step(_ConvTransposeNd_Block):
    def __init__(self, in_f, out_f):
        super(ConvTranspose3d_Block_1step, self).__init__()
        
        self.add_module('unpool1', nn.MaxUnpool3d(kernel_size=(1,4,4)))
        self.add_module('deconv1', nn.ConvTranspose3d(in_channels= in_f,
                                          out_channels= out_f,
                                          kernel_size= 3,
                                          padding= 1, 
                                          dilation= (1,1,1)))
        self.add_module('relu1', nn.ReLU())

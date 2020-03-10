import torch
import torch.nn as nn

from lfads import LFADS_SingleSession_Net

class Conv3d_LFADS_Net(nn.Module):
    def __init__(self, channel_dims = (16, 32), 
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
        
        self.device= device
        
        self.channel_dims = (1,) + channel_dims
        self.conv_layers = nn.ModuleList()
        
        for n in range(1, len(self.channel_dims)):
            self.conv_layers.add_module('{}{}'.format('conv', n),
                                        conv_block(in_f = self.channel_dims[n-1],
                                                   out_f= self.channel_dims[n]))
        
        self.deconv_layers = nn.ModuleList()
        for n in reversed(range(0, len(self.channel_dims))):
            self.deconv_layers.add_module('{}{}'.format('deconv', n),
                                          deconv_block(in_f = self.channel_dims[n],
                                                       out_f= self.channel_dims[n-1]))
            
            
        # Placeholder
        self.conv_output_size = self.channel_dims[-1] ** 3
        self.conv_dense_size = conv_dense_size
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.conv_dense = nn.Linear(in_features= self.conv_output_size,
                                    out_features= self.conv_dense_size)
        
        self.lfads = LFADS_Net(input_size= self.conv_dense_size,
                               g_encoder_size=g_encoder_size,
                               c_encoder_size=c_encoder_size,
                               g_latent_size=g_latent_size,
                               u_latent_size=u_latent_size,
                               controller_size=controller_size,
                               generator_size=generator_size,
                               factor_size=factor_size,
                               prior=prior,
                               clip_val=clip_val,
                               dropout=lfads_dropout,
                               max_norm=max_norm,
                               do_normalize_factors=do_normalize_factors,
                               factor_bias=factor_bias,
                               device= self.device)
        
    def forward(self, x):
        frame_per_block = 10
        x = video
        batch_size, num_ch, seq_len, w, h = x.shape
        num_blocks = int(seq_len/frame_per_block)
        
        x = x.view(batch_size,num_ch,num_blocks,frame_per_block,w,h).contiguous()
        x = x.permute(0,2,1,3,4,5).contiguous()
        x = x.view(batch_size * num_blocks,num_ch,frame_per_block,w,h).contiguous()
        

        Ind = list()
        for n, layer in enumerate(self.convlayers):
            x, ind1 = layer(x)
            Ind.append(ind1)
        
        num_out_ch = x.shape[1]
        w_out = x.shape[3]
        h_out = x.shape[4]
        x = x.view(batch_size,num_blocks,num_out_ch,frame_per_block,w_out,h_out).contiguous()
        x = x.permute(0,2,1,3,4,5).contiguous()
        
        x = x.view(batch_size,num_out_ch,seq_len,w_out,h_out).contiguous()

        
        x = x.permute(0,2,1,3,4)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        
        x = x.permute(1,0,2)
        factors, gen_inputs = self.lfads(x)
        x = factors
        x = x.permute(1,0,2)

        # call LFADS here:
        # x should be reshaped for LFADS [time x batch x cells]:
        # 
        # LFADS output should be also reshaped back for the conv decoder
        
        x = x.reshape(x.shape[0],x.shape[1],self.final_f,self.final_size, self.final_size)
        x = x.permute(0,2,1,3,4)
        
        x = x.view(batch_size,num_out_ch,num_blocks,frame_per_block,w_out,h_out).contiguous()
        x = x.permute(0,2,1,3,4,5).contiguous()
        x = x.view(batch_size * num_blocks,num_out_ch,frame_per_block,w_out,h_out).contiguous()

        for n, layer in enumerate(self.deconvlayers):     
            x = layer(x,Ind[self.n_layers-n-1])
        
        x = x.view(batch_size,num_blocks,1,frame_per_block,w,h).contiguous()
        x = x.permute(0,2,1,3,4,5)
        x = x.view(batch_size,1,seq_len,w,h)

        return x, factors

class Conv3d_Block(nn.Module):
    def __init__(self, in_f, out_f):
        super(Conv3d_Block, self).__init__()
        
        self.conv1 = nn.Conv3d(in_f, out_f, 
                               kernel_size=3, 
                               padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels= in_f,
                               out_channels= out_f, 
                               kernel_size=3, 
                               padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2),
                                  return_indices=True)
        self.relu2 = nn.ReLU()
        
    def forward(self,x):
        
        x = self.conv1(x)
        x, ind = self.pool1(x)
        x = self.relu1(x)
        
        return x, ind
    
class ConvTranspose3d_Block(nn.Module):
    def __init__(self, in_f, out_f):
        super(ConvTranspose3d_Block, self).__init__()
        
        self.unpool1 = nn.MaxUnpool3d(kernel_size=(1,2,2))
        
        self.deconv1 = nn.ConvTranspose3d(in_channels=in_f,
                                          out_channels=out_f,
                                          kernel_size=3,
                                          padding=1)
        self.relu1 = nn.ReLU()
        
    def forward(self, x, ind):
        
        x = self.unpool1(x,ind)
        x = self.deconv1(x)
        x = self.relu1(x)
        
        return x
#!/usr/bin/env python
# coding: utf-8


import os
import sys
import numpy as np
import argparse
import time
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

from lfads import LFADS_Net
from objective import *
from scheduler import LFADS_Scheduler


parser = argparse.ArgumentParser()
parser.add_argument('--save_loc', default='./', type=str)
parser.add_argument('--num_epochs', default=500, type=int)

global args; args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

torch.backends.cudnn.benchmark = True

class conv_block(nn.Module):# *args, **kwargs 
    def __init__(self, in_f, out_f):
        super(conv_block,self).__init__()
        
        self.conv1 = nn.Conv3d(in_f, out_f, 
                  kernel_size=3, 
                  padding=1,
                  dilation = (1,1,1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_f, out_f, 
                  kernel_size=3, 
                  padding=1,
                  dilation = 1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,4,4),
                     return_indices=True)
        self.relu2 = nn.ReLU()
        
    def forward(self,x):
        
        x = self.conv1(x)
        x, ind = self.pool1(x)
        x = self.relu1(x)
        
        return x, ind
        


class deconv_block(nn.Module):
    def __init__(self, in_f, out_f):
        super(deconv_block,self).__init__()
        
        self.unpool1 = nn.MaxUnpool3d(kernel_size=(1,4,4))
        
        self.deconv1 = nn.ConvTranspose3d(in_channels=in_f,
                                          out_channels=out_f,
                                          kernel_size=3,
                                          padding=1, 
                                          dilation = (1,1,1))
        self.relu1 = nn.ReLU()
        
    def forward(self,x,ind):
#         print(x.shape,ind.shape)
        x = self.unpool1(x,ind)
        x = self.deconv1(x)
        x = self.relu1(x)
        
        return x



class convVAE(nn.Module):
    def __init__(self):
        super(convVAE,self).__init__()
        
        device     = 'cuda' if torch.cuda.is_available() else 'cpu';
        
        in_f = 1
        out_f = [16,32]#[16,32,64]#[1,1]#
        all_f = [in_f,*out_f]
        self.n_layers = 2#3
        
        self.video_dim_space = 128
        self.video_dim_time = 50
        self.final_size = 8#
        self.final_f = 32#64#20#3
        self.factor_size = 4 # LFADS number of factors
        self.bottleneck_size = 64
        
        self.convlayers = nn.ModuleList()
        for n in range(0,self.n_layers):
            self.convlayers.add_module('{}{}'.format('ce', n),conv_block(all_f[n], all_f[n+1]))
#         self.convlayers.add_module('ce1',conv_block(out_f1, out_f2))
        
        self.deconvlayers = nn.ModuleList()
        for n in range(0,self.n_layers):
            self.deconvlayers.add_module('{}{}'.format('dec', n),deconv_block(all_f[self.n_layers-n], all_f[self.n_layers-n-1]))
#         self.deconvlayers.add_module('dec0',deconv_block(out_f2,out_f1))
#         self.deconvlayers.add_module('dec1',deconv_block(out_f1,in_f))
#         self.ce1 = conv_block(in_f, out_f1) 
#         self.ce2 = conv_block(out_f1, out_f2)

#         self.dec1 = deconv_block(out_f2,out_f1)
#         self.dec2 = deconv_block(out_f1,in_f) 
        self.fc1 = nn.Linear(self.final_size*self.final_size*self.final_f,self.bottleneck_size)
        self.lfads = LFADS_Net(self.bottleneck_size, factor_size = self.factor_size, #self.final_size * self.final_size * self.final_f
                 g_encoder_size  = 64, c_encoder_size = 0,
                 g_latent_size   = 64, u_latent_size  = 0,
                 controller_size = 0, generator_size = 64,
                 prior = {'g0' : {'mean' : {'value': 0.0, 'learnable' : True},
                                  'var'  : {'value': 0.1, 'learnable' : False}},
                          'u'  : {'mean' : {'value': 0.0, 'learnable' : False},
                                  'var'  : {'value': 0.1, 'learnable' : True},
                                  'tau'  : {'value': 10,  'learnable' : True}}},
                 clip_val=5.0, dropout=0.0, max_norm = 200, deep_freeze = False,
                 do_normalize_factors=True, device = device)
        self.fc2 = nn.Linear(self.factor_size,self.final_size*self.final_size*self.final_f)
        
    def forward(self,video):
        frame_per_block = 10
        x = video
        batch_size, num_ch, seq_len, w, h = x.shape
        num_blocks = int(seq_len/frame_per_block)
#         print(num_blocks)
        x = x.view(batch_size,num_ch,num_blocks,frame_per_block,w,h).contiguous()
        x = x.permute(0,2,1,3,4,5).contiguous()
        x = x.view(batch_size * num_blocks,num_ch,frame_per_block,w,h).contiguous()
        

        Ind = list()
#         conv_tic = time.time()
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
        x = self.fc1(x.view(batch_size,seq_len,w_out*h_out*num_out_ch))
#         conv_toc = time.time()
        
        x = x.permute(1,0,2)
#         lfads_tic = time.time()
#         r, (factors, gen_inputs) = self.lfads(x)
        (factors, gen_inputs) = self.lfads(x)
#         lfads_toc = time.time()
        x = factors
        x = x.permute(1,0,2)
        x = self.fc2(x)
        
#         print(conv_toc-conv_tic,lfads_toc - lfads_tic)
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
#         x, ind1 = self.ce0(video)
#         x, ind2 = self.ce1(x)
#         x = self.dec0(x,ind2)
#         v_p = self.dec1(x,ind1)
        

#         return v_p
        return x, factors


def get_data():
    
    from synthetic_data import generate_lorenz_data, SyntheticCalciumVideoDataset
    import numpy as np
    import torch
    from tempfile import mkdtemp
    import os.path as path
    import time

    filename_train = path.join(mkdtemp(), 'newfile_train.dat')
    filename_test = path.join(mkdtemp(), 'newfile_test.dat')
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    N_trials = 10
    N_inits = 35
    N_cells = 30
    N_steps = 100
    N_stepsinbin = 2
    
    data_dict = generate_lorenz_data(N_trials, N_inits, N_cells, N_steps, N_stepsinbin, save=False) # [N_trials, N_inits, N_cells, N_steps, N_stepsinbin]
    cells = data_dict['cells']
    traces = data_dict['train_fluor']
    train_data = SyntheticCalciumVideoDataset(traces=traces, cells=cells)
    test_data = SyntheticCalciumVideoDataset(traces=traces, cells=cells)
    
    num_workers = 0
    # how many samples per batch to load
    batch_size = 35

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    
    fp_train = np.memmap(filename_train, dtype='float32', mode='w+', shape=(batch_size,8,int(N_steps/N_stepsinbin),128,128))
    i = 0
    tic = time.time()
    for data in train_loader:
        videos = data
        fp_train[:,i,:,:,:] = np.squeeze(videos[:])
        i += 1
        
    fp_test = np.memmap(filename_test, dtype='float32', mode='w+', shape=(batch_size,8,int(N_steps/N_stepsinbin),128,128))
    i = 0
    tic = time.time()
    for data in test_loader:
        print(i)
        videos = data
        fp_test[:,i,:,:,:] = np.squeeze(videos[:])
        i += 1

    return fp_train, fp_test #train_data, train_loader, test_loader

class convLFADS_loss(nn.Module):
    def __init__(self,
                 kl_weight_init=0.1, l2_weight_init=0.1,
                 kl_weight_schedule_dur = 2000, l2_weight_schedule_dur = 2000,
                 kl_weight_schedule_start = 0, l2_weight_schedule_start = 0,
                 kl_weight_max=1.0, l2_weight_max=1.0,
                 l2_con_scale=1.0, l2_gen_scale=1.0):
        super(convLFADS_loss,self).__init__()
        
        self.loss_weights = {'kl' : {'weight' : kl_weight_init,
                                     'schedule_dur' : kl_weight_schedule_dur,
                                     'schedule_start' : kl_weight_schedule_start,
                                     'max' : kl_weight_max,
                                     'min' : kl_weight_init},
                             'l2' : {'weight' : l2_weight_init,
                                     'schedule_dur' : l2_weight_schedule_dur,
                                     'schedule_start' : l2_weight_schedule_start,
                                     'max' : l2_weight_max,
                                     'min' : l2_weight_init}}
        self.l2_con_scale = l2_con_scale
        self.l2_gen_scale = l2_gen_scale
        self.recon_loss = nn.MSELoss()
        
    def forward(self, video_orig, video_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        
#         recon_loss = -self.loglikelihood(x_orig.permute(1, 0, 2), x_recon['data'].permute(1, 0, 2))
        recon_loss = self.recon_loss(video_recon,video_orig)

        kl_loss = kl_weight * kldiv_gaussian_gaussian(post_mu  = model.g_posterior_mean,
                                                      post_lv  = model.g_posterior_logvar,
                                                      prior_mu = model.g_prior_mean,
                                                      prior_lv = model.g_prior_logvar)
    
        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.generator.gru_generator.hidden_weight_l2_norm()
    
#         if hasattr(model, 'controller'):
#             kl_loss += kl_weight * kldiv_gaussian_gaussian(post_mu  = model.u_posterior_mean,
#                                                            post_lv  = model.u_posterior_logvar,
#                                                            prior_mu = model.u_prior_mean,
#                                                            prior_lv = model.u_prior_logvar)
            
#             l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.controller.gru_controller.hidden_weight_l2_norm()
            
        return recon_loss, kl_loss, l2_loss
        
    
    
    
def train_convVAE(fp_train,fp_test,n_epochs): #model,
    
    device     = 'cuda' if torch.cuda.is_available() else 'cpu';
    model = convVAE().to(device)
    lfads = model.lfads.to(device)
    
    for ix, (name, param) in enumerate(model.named_parameters()):
        print(ix, name, list(param.shape), param.numel(), param.requires_grad)
#         total_params += param.numel()
        

    # number of epochs to train the model
#     n_epochs = 30
#     train_loader, test_loader = get_data()
#     model = convVAE()
#     criterion = nn.MSELoss()
    criterion = convLFADS_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     scheduler = LFADS_Scheduler(optimizer      =  optimizer,
#                                 mode           =  'min', 
#                                 factor         =  0.1, 
#                                 patience       =  10,
#                                 verbose        =  False, 
#                                 threshold      = 1e-4, 
#                                 threshold_mode = 'rel',
#                                 cooldown       =  0, 
#                                 min_lr         =  0,
#                                 eps            =  1e-8)
    
    writer_val = SummaryWriter(logdir=os.path.join(args.save_loc, 'log/val')) #
    writer_train = SummaryWriter(logdir=os.path.join(args.save_loc, 'log/train')) #
    
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        bw_tic = time.time()
        
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        i = 0
#         tic_train = time.time()
        num_step = fp_train.shape[1]
        videos = torch.zeros((35,1,50,128,128)).to(device)
        for i in range(num_step):#for data in train_loader:
            
            # _ stands in for labels, here
            # no need to flatten images
#             videos = data.to(device)
            videos[:,0,:,:,:] = torch.Tensor(fp_train[:,i,:,:,:]).to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            tic_forw = time.time()
            outputs,_ = model(videos)
            toc_forw = time.time()
#             print(toc_forw - tic_forw)
            # calculate the loss
            recon_loss, kl_loss, l2_loss = criterion(outputs, videos,lfads)
            loss = recon_loss + kl_loss + l2_loss
            # backward pass: compute gradient of the loss with respect to model parameters
            tic_backw = time.time()
            loss.backward()
            toc_backw = time.time()
#             print(toc_forw - tic_forw,toc_backw - tic_backw)
            # perform a single optimization step (parameter update)
            tic_optim = time.time()
            optimizer.step()
            toc_optim = time.time()
#             print(toc_optim - tic_optim)
            # update running training loss
            train_loss += loss.item()*videos.size(0)
            i += 1
#         toc_train = time.time()
#         print(toc_train - tic_train)   
        writer_train.add_scalar('total/loss', train_loss, epoch)
        
        test_loss = 0.0
#         tic_val = time.time()
        with torch.no_grad():
            num_step = fp_test.shape[1]
            videos_test = torch.zeros((35,1,50,128,128)).to(device)
            for j in range(num_step):#for data_test in test_loader:

    #             print(i)
                # _ stands in for labels, here
                # no need to flatten images
#                 videos_test = data_test.to(device)
                videos_test[:,0,:,:,:] = torch.Tensor(fp_test[:,j,:,:,:]).to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs_test,_ = model(videos_test)
                # calculate the loss
                recon_loss_test, kl_loss_test, l2_loss_test = criterion(outputs_test, videos_test,lfads)
                loss_test = recon_loss_test + kl_loss_test + l2_loss_test

                # update running training loss
                test_loss += loss_test.item()*videos.size(0)
                i += 1
#             toc_val = time.time()
#         print(toc_val - tic_val)
        writer_val.add_scalar('total/loss', test_loss, epoch)
            
#             scheduler.step(loss)
            
#         print avg training statistics  
        train_loss = train_loss/len(fp_train)
        test_loss = test_loss/len(fp_test)
        print(len(fp_train))
        bw_toc = time.time()
        print('Epoch: {} \tTotal Loss: {:.6f} \tl2 Loss: {:.6f} \tkl Loss: {:.6f} \tTest Loss {:.6f} \tTime {:.3f} s'.format(
            epoch, train_loss, l2_loss, kl_loss, test_loss, (bw_toc - bw_tic)))
#     print(bw_toc - bw_tic)    
    #save the trained model
    torch.save(model, os.path.join(args.save_loc, 'entire_model.pth'))

if __name__=="__main__":
#     train_data, train_loader, test_loader = get_data()
    fp_train, fp_test = get_data()
    train_convVAE(fp_train,fp_test,args.num_epochs) #


import torch
import torch.nn.functional as F
import torchvision
import time
import os
import pdb
import functools, collections, operator


class RunManager():
    def __init__(self, model, objective, optimizer, scheduler,
                 train_dl, valid_dl, transforms,
                 plotter=None, writer=None, do_health_check=False,
                 max_epochs=1000, save_loc = '/tmp/', load_checkpoint=False):
    
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu';
        self.model      = model
        self.objective  = objective
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.train_dl   = train_dl
        self.valid_dl   = valid_dl
        self.transforms = transforms
        self.writer     = writer
        self.plotter    = plotter
            
        self.max_epochs      = max_epochs
        self.do_health_check = do_health_check
        self.save_loc        = save_loc
        
        self.epoch = 0
        self.step  = 0
        self.best  = float('inf')
        
        self.loss_dict = {'train' : {},
                          'valid' : {},
                          'l2'    : []}
        
        if load_checkpoint:
            self.load_checkpoint('recent')
            
    def run(self):  
        for epoch in range(self.epoch, self.max_epochs):
            if self.optimizer.param_groups[0]['lr'] < self.scheduler.min_lrs[0]:
                break
            self.epoch = epoch + 1
            tic = time.time()
            loss_dict_list = []
            
            self.model.train()
            for i,x in enumerate(self.train_dl):
#                 print(x[0].session)
                x = x[0]
                self.optimizer.zero_grad()
                recon, latent = self.model(x)
                loss, loss_dict = self.objective(x_orig= x,
                                                 x_recon= recon,
                                                 model= self.model)
                
                loss_dict_list.append(loss_dict)
                

                bw_tic = time.time()
                loss.backward()
                bw_toc = time.time()
                
                if torch.isnan(loss.data):
                    break
                
                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.model.max_norm)
            
                # update the weights
                self.optimizer.step()
                
                self.objective.weight_schedule_fn(self.step)
                
                if self.model.do_normalize_factors:
                    # Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
                    self.model.normalize_factors()
                    
                if self.model.deep_freeze:
                    self.optimizer, self.scheduler = self.model.unfreeze_parameters(self.step, self.optimizer, self.scheduler)
                    
                self.step += 1
                
            if torch.isnan(loss.data):
                break
              
            train_data = x.clone()
            loss_dict = {} 
            for d in loss_dict_list: 
                for k in d.keys(): 
                    loss_dict[k] = loss_dict.get(k, 0) + d[k]/len(loss_dict_list)
            for key, val in loss_dict.items():
                if key in self.loss_dict['train'].keys():
                    self.loss_dict['train'][key].append(loss_dict[key])
                elif key == 'l2':
                    self.loss_dict[key].append(loss_dict[key])
                else:
                    self.loss_dict['train'][key] = [loss_dict[key]]
            
            self.scheduler.step(self.loss_dict['train']['total'][-1])


            loss_dict_list = []
            self.model.eval()
            for i, x in enumerate(self.valid_dl):
                with torch.no_grad():
                    x = x[0]
                    recon, latent = self.model(x)
                    loss, loss_dict = self.objective(x_orig= x, x_recon= recon, model= self.model)
                    loss_dict_list.append(loss_dict)
                    
            valid_data = x.clone()
            loss_dict = {} 
            for d in loss_dict_list: 
                for k in d.keys(): 
                    loss_dict[k] = loss_dict.get(k, 0) + d[k]/len(loss_dict_list)

            for key, val in loss_dict.items():
                if key in self.loss_dict['valid'].keys():
                    self.loss_dict['valid'][key].append(loss_dict[key])
                elif key == 'l2':
                    pass
                else:
                    self.loss_dict['valid'][key] = [loss_dict[key]]
                    
            valid_loss = self.loss_dict['valid']['total'][-1]
            if valid_loss < self.best:
                self.best = 0
                for key,val in self.loss_dict['valid'].items():
                    if 'recon' in key:
                        self.best += val[-1]
                    if ('kl' in key or 'l2' in key):
                        full_val = val[-1] / self.objective.loss_weights[key]['weight']
                        self.best += full_val
                self.save_checkpoint('best')
                
            self.save_checkpoint()
            if self.writer is not None:
                self.write_to_tensorboard()
                if self.plotter is not None:
                    if self.epoch % 25 == 0:
                        self.plot_to_tensorboard(train_data, valid_data)
                        
                if self.do_health_check:
                    self.health_check(self.model)
                    
            toc = time.time()
            
            results_string = 'Epoch %5d, Epoch time = %.3f s, Loss (train, valid): '%(self.epoch, toc - tic)
            for key in self.loss_dict['train'].keys():
                train_loss = self.loss_dict['train'][key][self.epoch-1]
                valid_loss = self.loss_dict['valid'][key][self.epoch-1]
                results_string+= ' %s (%.2f, %.2f),'%(key, train_loss, valid_loss)
            
            results_string+= ' %s (%.2f)'%('l2', self.loss_dict['l2'][self.epoch-1])
            
            print(results_string, flush=True)
            
    def write_to_tensorboard(self):
        
        # Write loss to tensorboard
        
        for ix, key in enumerate(self.loss_dict['train'].keys()):
            train_loss = self.loss_dict['train'][key][self.epoch-1]
            valid_loss = self.loss_dict['valid'][key][self.epoch-1]
            
            self.writer.add_scalars('1_Loss/%i_%s'%(ix+1, key), {'Training' : float(train_loss),
                                                                 'Validation' : float(valid_loss)}, self.epoch)
        l2_loss = self.loss_dict['l2'][self.epoch-1]
        self.writer.add_scalar('1_Loss/4_L2_loss', float(l2_loss), self.epoch)

        for jx, grp in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar('2_Optimizer/1.%i_Learning_Rate_PG'%(jx+1), grp['lr'], self.epoch)
        
        for kx, key in enumerate(self.objective.loss_weights.keys()):
            weight = self.objective.loss_weights[key]['weight']
            self.writer.add_scalar('2_Optimizer/2.%i_%s_weight'%(kx+1, key), weight, self.epoch)
        
    def plot_to_tensorboard(self, train_data, valid_data):
        figs_dict_train = self.plotter['train'].plot_summary(model = self.model, data  = train_data)
        
        figs_dict_valid = self.plotter['valid'].plot_summary(model = self.model, data  = valid_data)
        
        fig_names = ['traces', 'inputs', 'factors', 'rates', 'spikes']
        for fn in fig_names:
            if fn in figs_dict_train.keys():
                self.writer.add_figure('%s/train'%(fn), figs_dict_train[fn], self.epoch, close=True)
            elif 'truth_%s'%fn in figs_dict_train.keys():
                self.writer.add_figure('%s/train'%(fn), figs_dict_train['truth_%s'%fn], self.epoch, close=True)

            if fn in figs_dict_valid.keys():
                self.writer.add_figure('%s/valid'%(fn), figs_dict_valid[fn], self.epoch, close=True)
            elif 'truth_%s'%fn in figs_dict_valid.keys():
                self.writer.add_figure('%s/valid'%(fn), figs_dict_valid['truth_%s'%fn], self.epoch, close=True)
            
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    def health_check(self, model):
        '''
        Gets gradient norms for each parameter and writes to tensorboard
        '''
        
        for ix, (name, param) in enumerate(model.named_parameters()):
            if param.grad is not None:
                self.writer.add_scalar('3_Gradient_norms/%i_%s'%(ix, name), param.grad.data.norm(), self.epoch)
            else:
                self.writer.add_scalar('3_Gradient_norms/%i_%s'%(ix, name), 0.0, self.epoch)
                
            if 'weight' in name:
                self.writer.add_scalar('4_Weight_norms/%i_%s'%(ix, name), param.data.norm(), self.epoch)
        
    def save_checkpoint(self, output_filename='recent'):
                # Create dictionary of training variables
        train_dict = {'best' : self.best,
                      'loss_dict': self.loss_dict,
                      'loss_weights' : self.objective.loss_weights,
                      'epoch' : self.epoch, 'step' : self.step}
        
        # Save network parameters, optimizer state, and training variables
        if not os.path.isdir(self.save_loc+'checkpoints/'):
            os.mkdir(self.save_loc+'checkpoints/')
        
        torch.save({'net' : self.model.state_dict(), 'opt' : self.optimizer.state_dict(),
                    'sched': self.scheduler.state_dict(), 'run_manager' : train_dict},
                     self.save_loc+'checkpoints/' + output_filename + '.pth')
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
        
    def load_checkpoint(self, input_filename='recent'):
        if os.path.exists(self.save_loc + 'checkpoints/' + input_filename + '.pth'):
            state_dict = torch.load(self.save_loc + 'checkpoints/' + input_filename + '.pth')
            self.model.load_state_dict(state_dict['net'])
            if len(state_dict['opt']['param_groups']) > 1:
                self.optimizer, self.scheduler = self.model.unfreeze_parameters(state_dict['run_manager']['step'], self.optimizer, self.scheduler)
            self.optimizer.load_state_dict(state_dict['opt'])
            self.scheduler.load_state_dict(state_dict['sched'])

            self.best = state_dict['run_manager']['best']
            self.loss_dict = state_dict['run_manager']['loss_dict']
            self.objective.loss_weights = state_dict['run_manager']['loss_weights']
            self.epoch = state_dict['run_manager']['epoch']
            self.step  = state_dict['run_manager']['step']
            

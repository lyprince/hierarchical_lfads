import torch
import torch.nn.functional as F
import torchvision
import time
import os
import pdb

class RunManager():
    def __init__(self, model, objective, optimizer, scheduler,
                 train_dl, valid_dl, plotter=None, writer=None, max_epochs=1000,
                 save_loc = '/tmp/', load_checkpoint=False, do_health_check=False):
    
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu';
        self.model      = model
        self.objective  = objective
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.train_dl   = train_dl
        self.valid_dl   = valid_dl
        self.writer     = writer
        self.plotter    = plotter
            
        self.max_epochs   = max_epochs
        self.do_health_check = do_health_check
        self.save_loc     = save_loc
        
        self.epoch = 0
        self.step  = 0
        self.best  = float('inf')
        
        self.loss_dict = {'train' : {'total' : [],
                                      'recon' : [],
                                      'kl'    : []},
                          'valid' : {'total' : [],
                                      'recon' : [],
                                      'kl'    : []},
                          'l2' : []}
        
        if load_checkpoint:
            self.load_checkpoint('recent')
            
    def run(self):
        for epoch in range(self.epoch, self.max_epochs):
            if self.optimizer.param_groups[0]['lr'] < self.scheduler.min_lrs[0]:
                break
            self.epoch = epoch + 1
            tic = time.time()
            train_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            epoch_l2_loss = 0
            self.model.train()
            for i,x in enumerate(self.train_dl):
                x = x[0].permute(1, 0, 2)
                self.optimizer.zero_grad()
                recon, factors = self.model(x)
                recon_loss, kl_loss, l2_loss = self.objective(x_orig= x, x_recon= recon['data'], model= self.model)
                
                loss = recon_loss + kl_loss + l2_loss
                train_loss += float(loss.data)
                train_recon_loss += float(recon_loss.data)
                train_kl_loss += float(kl_loss.data)
                epoch_l2_loss += float(l2_loss.data)
#                 print('Loss = %.3f'%float(loss.data))
                bw_tic = time.time()
                loss.backward()
                bw_toc = time.time()
#                 print('Backward took %.3f ms'%(1000*(bw_toc - bw_tic)))
               
                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.model.max_norm)
            
                # update the weights
                self.optimizer.step()
                
                self.objective.weight_schedule_fn(self.step)
                
                if self.model.normalize_factors:
                    # Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
                    self.model.generator.fc_factors.weight.data = F.normalize(self.model.generator.fc_factors.weight.data, dim=1)
                    
                self.step += 1
            
            train_loss /= (i + 1)
            train_recon_loss /= (i + 1)
            train_kl_loss /= (i + 1)
            epoch_l2_loss /= (i + 1)
            
            self.loss_dict['train']['total'].append(train_loss)
            self.loss_dict['train']['recon'].append(train_recon_loss)
            self.loss_dict['train']['kl'].append(train_kl_loss)
            self.loss_dict['l2'].append(epoch_l2_loss)
            
            self.scheduler.step(train_loss)
            

            valid_loss = 0
            valid_recon_loss = 0
            valid_kl_loss = 0
            self.model.eval()
            for i, x in enumerate(self.valid_dl):
                with torch.no_grad():
                    x = x[0].permute(1, 0, 2)
                    recon, factors = self.model(x)
                    recon_loss, kl_loss, l2_loss = self.objective(x_orig= x, x_recon= recon['data'], model= self.model)
                    
                    loss = recon_loss + kl_loss + l2_loss
                    valid_loss += float(loss.data)
                    valid_recon_loss += float(recon_loss.data)
                    valid_kl_loss += float(kl_loss.data)
                    
                    
            valid_loss /= (i + 1)
            valid_recon_loss /= (i+1)
            valid_kl_loss /= (i+1)
            self.loss_dict['valid']['total'].append(valid_loss)
            self.loss_dict['valid']['recon'].append(valid_recon_loss)
            self.loss_dict['valid']['kl'].append(valid_kl_loss)
            if valid_loss < self.best:
                self.best = valid_loss
                self.save_checkpoint('best')
                
            self.save_checkpoint()
            if self.writer is not None:
                self.write_to_tensorboard()
                if self.plotter is not None:
                    if self.epoch % 25 == 0:
                        self.plot_to_tensorboard()
                        
                if self.do_health_check:
                    self.health_check(self.model)
            
            
            toc = time.time()
            print('Epoch %4d, Epoch time = %.3f s, Loss [w] (train, valid): Total (%.3f, %.3f), Recon (%.2f, %.2f), KL [%.2f] (%.2f, %.2f), L2 [%.2f] (%.2f)'%(self.epoch, toc - tic, train_loss, valid_loss, train_recon_loss, valid_recon_loss, self.objective.loss_weights['kl']['weight'], train_kl_loss, valid_kl_loss, self.objective.loss_weights['l2']['weight'], epoch_l2_loss))
        
#         self.writer.add_graph(self.model, x)
            
    def write_to_tensorboard(self):
            
        # Write loss to full_loss_store dict
        train_loss       = self.loss_dict['train']['total'][self.epoch-1]
        train_recon_loss = self.loss_dict['train']['recon'][self.epoch-1]
        train_kl_loss    = self.loss_dict['train']['kl'][self.epoch-1]

        valid_loss       = self.loss_dict['valid']['total'][self.epoch-1]
        valid_recon_loss = self.loss_dict['valid']['recon'][self.epoch-1]
        valid_kl_loss    = self.loss_dict['valid']['kl'][self.epoch-1]
        
        l2_loss = self.loss_dict['l2'][self.epoch-1]
        
        # Retrieve loss weights from cost_weights dict
        kl_weight = self.objective.loss_weights['kl']['weight']
        l2_weight = self.objective.loss_weights['l2']['weight']
        
        # Write loss to tensorboard
        self.writer.add_scalars('1_Loss/1_Total_Loss', {'Training'  : float(train_loss), 
                                                        'Validation': float(valid_loss)}, self.epoch)

        self.writer.add_scalars('1_Loss/2_Reconstruction_Loss', {'Training'  :  float(train_recon_loss), 
                                                                 'Validation': float(valid_recon_loss)}, self.epoch)

        self.writer.add_scalars('1_Loss/3_KL_Loss' , {'Training'  : float(train_kl_loss), 
                                                      'Validation': float(valid_kl_loss)}, self.epoch)

        self.writer.add_scalar('1_Loss/4_L2_loss', float(l2_loss), self.epoch)

        self.writer.add_scalar('2_Optimizer/1_Learning_Rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        self.writer.add_scalar('2_Optimizer/2_KL_weight', kl_weight, self.epoch)
        self.writer.add_scalar('2_Optimizer/3_L2_weight', l2_weight, self.epoch)
        
    def plot_to_tensorboard(self):
        figs_dict_train = self.plotter['train'].plot_summary(model = self.model,
                                                    data  = self.train_dl.dataset.tensors[0])
        
        figs_dict_valid = self.plotter['valid'].plot_summary(model = self.model,
                                                    data  = self.valid_dl.dataset.tensors[0])
        
        self.writer.add_figure('Examples/1_Train', figs_dict_train['traces'], self.epoch, close=True)
        if 'inputs' in figs_dict_train.keys():
            self.writer.add_figure('Inputs/1_Train', figs_dict_train['inputs'], self.epoch, close=True)

        self.writer.add_figure('Examples/2_Valid', figs_dict_valid['traces'], self.epoch, close=True)
        if 'inputs' in figs_dict_valid.keys():
            self.writer.add_figure('Inputs/2_Valid', figs_dict_valid['inputs'], self.epoch, close=True)

        if 'truth_factors' in figs_dict_train.keys():
            self.writer.add_figure('Ground_truth/1_Train/1_Factors', figs_dict_train['truth_factors'], self.epoch, close=True)
        else:
            self.writer.add_figure('Factors/1_Train', figs_dict_train['factors'], self.epoch, close=True)

        if 'truth_rates' in figs_dict_train.keys():
            self.writer.add_figure('Ground_truth/1_Train/2_Rates', figs_dict_train['truth_rates'], self.epoch, close=True)
        else:
            self.writer.add_figure('Rates/1_Train', figs_dict_train['rates'], self.epoch, close=True)

        if 'truth_factors' in figs_dict_valid.keys():
            self.writer.add_figure('Ground_truth/2_Valid/1_Factors', figs_dict_valid['truth_factors'], self.epoch, close=True)
        else:
            self.writer.add_figure('Factors/2_Valid', figs_dict_valid['factors'], self.epoch, close=True)

        if 'truth_rates' in figs_dict_valid.keys():
            self.writer.add_figure('Ground_truth/2_Valid/2_Rates', figs_dict_valid['truth_rates'], self.epoch, close=True)
        else:
            self.writer.add_figure('Rates/2_Valid', figs_dict_valid['rates'], self.epoch, close = True)
            
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
        
    def load_checkpoint(self, input_filename='recent'):
        if os.path.exists(self.save_loc + 'checkpoints/' + input_filename + '.pth'):
            state_dict = torch.load(self.save_loc + 'checkpoints/' + input_filename + '.pth')
            self.model.load_state_dict(state_dict['net'])
            self.optimizer.load_state_dict(state_dict['opt'])
            self.scheduler.load_state_dict(state_dict['sched'])

            self.best = state_dict['run_manager']['best']
            self.loss_dict = state_dict['run_manager']['loss_dict']
            self.objective.loss_weights = state_dict['run_manager']['loss_weights']
            self.epoch = state_dict['run_manager']['epoch']
            self.step  = state_dict['run_manager']['step']
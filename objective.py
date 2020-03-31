import torch
import torch.nn as nn
import pdb
from math import log

class Base_Loss(nn.Module):
    def __init__(self, loss_weight_dict, l2_gen_scale=0.0, l2_con_scale=0.0):
        super(Base_Loss, self).__init__()
        self.loss_weights = loss_weight_dict
        self.l2_gen_scale = l2_gen_scale
        self.l2_con_scale = l2_con_scale

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
        
    def forward(self, x_orig, x_recon, model):
        pass
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    def weight_schedule_fn(self, step):
        '''
        weight_schedule_fn(step)
        
        Calculate the KL and L2 regularization weights from the current training step number. Imposes
        linearly increasing schedule on regularization weights to prevent early pathological minimization
        of KL divergence and L2 norm before sufficient data reconstruction improvement. See bullet-point
        4 of section 1.9 in online methods
        
        required arguments:
            - step (int) : training step number
        '''
        
        for key in self.loss_weights.keys():
            # Get step number of scheduler
            weight_step = max(step - self.loss_weights[key]['schedule_start'], 0)
            
            # Calculate schedule weight
            self.loss_weights[key]['weight'] = max(min(self.loss_weights[key]['max'] * weight_step/ self.loss_weights[key]['schedule_dur'], self.loss_weights[key]['max']), self.loss_weights[key]['min'])

    def any_zero_weights(self):
        for key, val in self.loss_weights.items():
            if val['weight'] == 0:
                return True
            else:
                pass
        return False

class SVLAE_Loss(Base_Loss):
    def __init__(self, loglikelihood_obs, loglikelihood_deep,
                 loss_weight_dict = {'kl_obs' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0,    'max' : 1.0, 'min' : 0.0},
                                     'kl_deep': {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 2000, 'max' : 1.0, 'min' : 0.0},
                                     'l2'     : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 2000, 'max' : 1.0, 'min' : 0.0},
                                     'recon_deep' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 2000, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(SVLAE_Loss, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood_obs  = loglikelihood_obs
        self.loglikelihood_deep = loglikelihood_deep

    def forward(self, x_orig, x_recon, model):
        kl_obs_weight = self.loss_weights['kl_obs']['weight']
        kl_deep_weight = self.loss_weights['kl_deep']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        recon_deep_weight = self.loss_weights['recon_deep']['weight']
#         pdb.set_trace()

        recon_obs_loss  = -self.loglikelihood_obs(x_orig, x_recon['data'], model.obs_model.generator.calcium_generator.logvar)
        recon_deep_loss = -self.loglikelihood_deep(x_recon['spikes'].permute(1, 0, 2), x_recon['rates'].permute(1, 0, 2))
        recon_deep_loss = recon_deep_weight * recon_deep_loss

        kl_obs_loss = kl_obs_weight * kldiv_gaussian_gaussian(post_mu   = model.obs_model.u_posterior_mean,
                                                               post_lv  = model.obs_model.u_posterior_logvar,
                                                               prior_mu = model.obs_model.u_prior_mean,
                                                               prior_lv = model.obs_model.u_prior_logvar)

        kl_deep_loss = kl_deep_weight * kldiv_gaussian_gaussian(post_mu  = model.deep_model.g_posterior_mean,
                                                                post_lv  = model.deep_model.g_posterior_logvar,
                                                                prior_mu = model.deep_model.g_prior_mean,
                                                                prior_lv = model.deep_model.g_prior_logvar)

        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.deep_model.generator.gru_generator.hidden_weight_l2_norm()

        if hasattr(model.deep_model, 'controller'):
            kl_deep_loss += kl_deep_weight * kldiv_gaussian_gaussian(post_mu  = model.deep_model.u_posterior_mean,
                                                                     post_lv  = model.deep_model.u_posterior_logvar,
                                                                     prior_mu = model.deep_model.u_prior_mean,
                                                                     prior_lv = model.deep_model.u_prior_logvar)

            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.deep_model.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = recon_obs_loss + recon_deep_loss +  kl_obs_loss + kl_deep_loss + l2_loss
        loss_dict = {'recon_obs'  : float(recon_obs_loss.data),
                     'recon_deep' : float(recon_deep_loss.data),
                     'kl_obs'     : float(kl_obs_loss.data),
                     'kl_deep'    : float(kl_deep_loss.data),
                     'l2'         : float(l2_loss.data),
                     'total'      : float(loss.data)}

        return loss, loss_dict

class LFADS_Loss(Base_Loss):
    def __init__(self, loglikelihood,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(LFADS_Loss, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood = loglikelihood
        
    def forward(self, x_orig, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        
        recon_loss = -self.loglikelihood(x_orig, x_recon['data'])

        kl_loss = kl_weight * kldiv_gaussian_gaussian(post_mu  = model.g_posterior_mean,
                                                      post_lv  = model.g_posterior_logvar,
                                                      prior_mu = model.g_prior_mean,
                                                      prior_lv = model.g_prior_logvar)
    
        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.generator.gru_generator.hidden_weight_l2_norm()
    
        if hasattr(model, 'controller'):
            kl_loss += kl_weight * kldiv_gaussian_gaussian(post_mu  = model.u_posterior_mean,
                                                           post_lv  = model.u_posterior_logvar,
                                                           prior_mu = model.u_prior_mean,
                                                           prior_lv = model.u_prior_logvar)
            
            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = recon_loss +  kl_loss + l2_loss
        loss_dict = {'recon' : float(recon_loss.data),
                     'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data),
                     'total' : float(loss.data)}

        return loss, loss_dict
    
class Conv_LFADS_Loss(LFADS_Loss):
    
    def __init__(self, loglikelihood,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(Conv_LFADS_Loss, self).__init__(loglikelihood=loglikelihood,
                                              loss_weight_dict=loss_weight_dict,
                                              l2_con_scale=l2_con_scale,
                                              l2_gen_scale=l2_gen_scale)
        
        
    def forward(self, x_orig, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        recon_loss = -self.loglikelihood(x_orig, x_recon['data'])

        kl_loss = kl_weight * kldiv_gaussian_gaussian(post_mu  = model.lfads.g_posterior_mean.to(torch.float32),
                                                      post_lv  = model.lfads.g_posterior_logvar.to(torch.float32),
                                                      prior_mu = model.lfads.g_prior_mean.to(torch.float32),
                                                      prior_lv = model.lfads.g_prior_logvar.to(torch.float32)).to(dtype=x_orig.dtype)
    
        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.lfads.generator.gru_generator.hidden_weight_l2_norm()
    
        if hasattr(model, 'controller'):
            kl_loss += kl_weight * kldiv_gaussian_gaussian(post_mu  = model.lfads.u_posterior_mean.to(torch.float32),
                                                           post_lv  = model.lfads.u_posterior_logvar.to(torch.float32),
                                                           prior_mu = model.lfads.u_prior_mean.to(torch.float32),
                                                           prior_lv = model.lfads.u_prior_logvar.to(torch.float32)).to(dtype=x_orig.dtype7)
            
            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.lfads.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = recon_loss +  kl_loss + l2_loss
        loss_dict = {'recon' : float(recon_loss.data),
                     'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data),
                     'total' : float(loss.data)}

        return loss, loss_dict
        
class LogLikelihoodPoisson(nn.Module):
    
    def __init__(self, dt=1.0, device='cpu'):
        super(LogLikelihoodPoisson, self).__init__()
        self.dt = dt
        
    def forward(self, k, lam):
#         pdb.set_trace()
        return loglikelihood_poisson(k, lam*self.dt)

class LogLikelihoodPoissonSimple(nn.Module):
    
    def __init__(self, dt=1.0, device='cpu'):
        super(LogLikelihoodPoissonSimple, self).__init__()
        self.dt = dt
    
    def forward(self, k, lam):
        return loglikelihood_poissonsimple(k, lam*self.dt)

class LogLikelihoodPoissonSimplePlusL1(nn.Module):
    
    def __init__(self, dt=1.0, device='cpu'):
        super(LogLikelihoodPoissonSimplePlusL1, self).__init__()
        self.dt = dt
    
    def forward(self, k, lam):
        return loglikelihood_poissonsimple_plusl1(k, lam*self.dt)
    
def loglikelihood_poisson(k, lam):
    '''
    loglikelihood_poisson(k, lam)

    Log-likelihood of Poisson distributed counts k given intensity lam.

    Arguments:
        - k (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - lam (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
    '''
    return (k * torch.log(lam) - lam - torch.lgamma(k + 1)).mean(dim=0).sum()

def loglikelihood_poissonsimple_plusl1(k, lam):
    return (k * torch.log(lam) - lam - torch.abs(k)).mean(dim=0).sum()

def loglikelihood_poissonsimple(k, lam):
    return (k * torch.log(lam) - lam).mean(dim=0).sum()

class LogLikelihoodGaussian(nn.Module):
    def __init__(self):
        super(LogLikelihoodGaussian, self).__init__()
        
    def forward(self, x, mean, logvar=None):
        if logvar:
            return loglikelihood_gaussian(x, mean, logvar)
        else:
            return torch.nn.functional.mse_loss(x, mean, reduction='sum')/x.shape[0]
    
def loglikelihood_gaussian(x, mean, logvar):
    from math import pi
    return -0.5*(log(2*pi) + logvar + ((x - mean).pow(2)/torch.exp(logvar))).mean(dim=0).sum()
        

def kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv):
    '''
    kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv)

    KL-Divergence between a prior and posterior diagonal Gaussian distribution.

    Arguments:
        - post_mu (torch.Tensor): mean for the posterior
        - post_lv (torch.Tensor): logvariance for the posterior
        - prior_mu (torch.Tensor): mean for the prior
        - prior_lv (torch.Tensor): logvariance for the prior
    '''
    klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
         + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).mean(dim=0).sum()
    return klc
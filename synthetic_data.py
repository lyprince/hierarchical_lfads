import numpy as np
from scipy import signal as sig
import pdb
import h5py
import torch
import torchvision
import tempfile

def euler_step(x, f, dt):
    return x + dt * f(x)

def rk4_step(x, f, dt):
    k1 = dt * f(x)
    k2 = dt * f(x + 0.5*k1)
    k3 = dt * f(x + 0.5*k2)
    k4 = dt * f(x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4)/6
    
class DynamicalSystem():
    def __init__(self):
        pass
        
    def gradient(self, state):
        pass

    
    def rescale(self, xt):
        return xt
    
    def generate_inputs(self, dims):
        return None
    
    def update(self, order=4):
        if order == 1:
            return euler_step(x=self.state, f=self.gradient, dt=self.dt)
        else:
            return rk4_step(x= self.state, f=self.gradient, dt=self.dt)
    
    def integrate(self, num_steps, inputs, burn_steps = 0):
        
        result = np.zeros((num_steps,) + self.state.shape)
        for t in range(burn_steps):
            self.state = self.update()
        for t in range(num_steps):
            self.state = self.update()
            if inputs is not None:
                self.state += inputs[t]
            result[t] = self.state
            
        result = self.rescale(result)
        self.result = result
        return result
    
class LorenzSystem(DynamicalSystem):
    def __init__(self, num_inits=100, weights=[10.0, 28.0, 8.0/3.0], dt=0.01):        
        self.state  = np.random.randn(num_inits, 3)
        self.weights = np.array(weights)
        self.num_inits = num_inits
        self.net_size = 3
        self.dt = dt
        
    def gradient(self, state):
        y1, y2, y3 = state.T
        w1, w2, w3 = self.weights
        dy1 = w1 * (y2 - y1)
        dy2 = y1 * (w2 - y3) - y2
        dy3 = y1 *  y2 - w3  * y3
        return np.array([dy1, dy2, dy3]).T
    
    def rescale(self, xt):
        xt -= xt.mean(axis=0).mean(axis=0)
        xt /= np.abs(xt).max()
        return xt
    
class EmbeddedLowDNetwork(DynamicalSystem):
    def __init__(self, low_d_system, net_size=64, base_rate=1.0, dt= 0.01):
        super(EmbeddedLowDNetwork, self).__init__()
        
        self.low_d_system = low_d_system
        self.net_size = net_size
        self.proj = (np.random.rand(self.low_d_system.net_size, self.net_size) + 1) * np.sign(np.random.randn(self.low_d_system.net_size, net_size))
        self.bias = np.log(base_rate)
        self.dt = dt
        self.num_inits = self.low_d_system.num_inits
        
    def gradient(self, state):
        return self.low_d_system.gradient(state)
    
    def rescale(self, xt):
        return np.exp(xt.dot(self.proj) + self.bias)
    
    def integrate(self, burn_steps, num_steps, inputs):
        result = self.low_d_system.integrate(burn_steps = burn_steps, num_steps = num_steps, inputs=inputs)
        result = self.rescale(result)
        self.result = result
        return result
        
class ChaoticNetwork(DynamicalSystem):
    
    def __init__(self, num_inits=100, max_rate = 50.0, net_size=64, weight_scale=5.0, dt= 0.01, inputs=None):
        self.dt = dt
        self.max_rate = max_rate
        self.num_inits = num_inits
        self.net_size = net_size
        
        self.state = np.random.randn(self.num_inits, self.net_size)
        self.weights = weight_scale * np.random.randn(self.net_size, self.net_size)/np.sqrt(self.net_size)
        
        self.inputs = inputs
        
    def gradient(self, state):
        return -state + np.tanh(state).dot(self.weights)
    
    def generate_inputs(self, dims):
        if self.inputs is not None:
            return self.inputs.generate(dims)
        else:
            return None
        
    def rescale(self, xt):
        return 0.5 * self.max_rate * (np.tanh(xt) + 1)
    
class RandomPerturbation():
    def __init__(self, t_span=[0.25, 0.75], scale = 20):
        self.t_span = t_span
        self.scale = scale
        
    def generate(self, dims):
        num_steps, num_trials, num_cells = dims
        u = np.zeros((num_steps, num_trials))
        perturb_step = np.random.randint(self.t_span[0]*num_steps, self.t_span[1]*num_steps, size=num_trials)
        u[perturb_step, list(range(num_trials))] += 1
        u = u[..., None] * np.random.randn(num_cells) * self.scale
        self.u = u
        return u
    
    def __getitem__(self, ix):
        return self.u[ix]
    
    def __len__(self):
        return(len(self.u))
        
        
class AR1Calcium(DynamicalSystem):
    
    def __init__(self, dims, tau=0.1, dt=0.01):
        self.state = np.zeros(dims)
        self.tau = tau
        self.dt = dt
        
    def gradient(self, state):
        return -state/self.tau
    
    def rescale(self, xt):
        return xt

class SyntheticCalciumDataGenerator():
    def __init__(self, system, seed, trainp = 0.8,
                 burn_steps = 1000, num_trials = 100, num_steps= 100,
                 tau_cal=0.1, dt_cal= 0.01, sigma=0.2,
                 frame_width=128, frame_height=128, cell_radius=4, save=True):
        
        self.seed = seed
        np.random.seed(seed)
        self.trainp = trainp
        
        self.system = system
        self.burn_steps = burn_steps
        
        self.num_steps  = num_steps
        self.num_trials = num_trials
        
        self.calcium_dynamics = AR1Calcium(dims=(self.num_trials,
                                                 self.system.num_inits,
                                                 self.system.net_size), 
                                           tau=tau_cal, dt=dt_cal)
        self.sigma = sigma
        
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.cell_radius = cell_radius
        
    def generate_dataset(self):
        inputs  = self.system.generate_inputs(dims=(self.num_steps, self.system.num_inits, self.system.net_size))
        rates   = self.system.integrate(burn_steps = self.burn_steps, num_steps = self.num_steps, inputs= inputs)
        if type(self.system) is EmbeddedLowDNetwork:
            latent = self.system.low_d_system.result
            latent = self.trials_repeat(latent)
        else:
            latent = None
        if inputs is not None:
            inputs = self.trials_repeat(inputs)
            
        rates   = self.trials_repeat(rates)
#         pdb.set_trace()
        spikes  = self.spikify(rates, self.calcium_dynamics.dt)
#         pdb.set_trace()
        calcium = self.calcium_dynamics.integrate(num_steps=self.num_steps, inputs=spikes.transpose(2, 0, 1, 3)).transpose(1, 2, 0, 3)
        fluor   = calcium + np.random.randn(*calcium.shape)*self.sigma
        
#         pdb.set_trace()
        cells, cell_loc = self.generate_cells(num_cells=self.system.net_size,
                                              frame_width=self.frame_width,
                                              frame_height=self.frame_height,
                                              cell_radius=self.cell_radius)
        
        data_dict = {}
        for data, data_name in zip((inputs, rates, latent, spikes, calcium, fluor), 
                                   ('inputs', 'rates', 'latent', 'spikes', 'calcium', 'fluor')):
            if data is not None:
                data_dict['train_%s'%data_name], data_dict['valid_%s'%data_name] = self.train_test_split(data)
        
        data_dict['cells'] = cells
        data_dict['cell_loc'] = cell_loc
        data_dict['dt'] = self.calcium_dynamics.dt
        
        return data_dict
        
    def trials_repeat(self, data):
        data = data[..., None] * np.ones(self.num_trials)
        return data.transpose(3, 1, 0, 2)
        
    def spikify(self, rates, dt):
        return np.random.poisson(rates*dt)
    
    def calcify(self, spikes):
        return self.calcium_dynamics.integrate(num_steps=num_steps, inputs=spikes)
        
    def generate_cells(self, num_cells, frame_width, frame_height, cell_radius):
        import skimage.draw as draw
        cell_loc = np.random.uniform(low=np.array([[0.0], [0.0]])* np.ones((1, num_cells)),
                                     high=np.array([[frame_width], [frame_height]]) * np.ones((1, num_cells)))
        A = np.zeros((num_cells, frame_width + 2*cell_radius, frame_height + 2*cell_radius))
        for ix in range(num_cells):
            r, c = cell_loc[:, ix]
            rr, cc = draw.circle(r, c, radius=cell_radius)
            A[ix, rr, cc] += 1
        return A[:, cell_radius:-cell_radius, cell_radius:-cell_radius], cell_loc
    
    def train_test_split(self, data):
        num_trials, num_inits, num_steps, num_cells = data.shape
        num_train = int(self.trainp * num_trials)
        train_data = data[:num_train].reshape(num_train*num_inits, num_steps, num_cells)
        valid_data = data[num_train:].reshape((num_trials - num_train)*num_inits, num_steps, num_cells)
        return train_data, valid_data
    

class ShenoyCalciumDataGenerator():
    def __init__(self, data_dir, trainp = 0.8, num_steps = None, num_trials = 2296, net_size = 202, tau_cal=0.1, dt_cal= 0.01,
                 sigma=0.2,frame_width=128, frame_height=128, cell_radius=4, save=True):
        
        
        
        self.data_dir = data_dir
        self.net_size = net_size
        self.trainp = trainp
                
        self.num_steps  = num_steps
        self.num_trials = num_trials
        self.net_size = net_size
        
        self.calcium_dynamics = AR1Calcium(dims=(self.num_trials,1,self.net_size), 
                                           tau=tau_cal, dt=dt_cal)
        self.sigma = sigma
        self.dt_cal = dt_cal
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.cell_radius = cell_radius
    
    def load_spike_data(self):
        
        f = h5py.File(self.data_dir,'r')
        unitspikesCount = f.get('unitspikesCount')
        trialVersion = f.get('trialVersion')
        trialType = f.get('trialType')

    
        return unitspikesCount, trialVersion, trialType
    
    def resample_data(self,X,num):
        
        num_samples = X.shape[2]
        bin_width = int(num_samples/num)
        print(np.sum(X[0,0,:,1]))
        Xds = np.sum(X.reshape(self.num_trials,1,-1, bin_width,self.net_size),3)
        
        return Xds
        
    
    
    
    def generate_dataset(self):
#         inputs  = self.system.generate_inputs(dims=(self.num_steps, self.system.num_inits, self.system.net_size))
#         rates   = self.system.integrate(burn_steps = self.burn_steps, num_steps = self.num_steps, inputs= inputs)
#         if type(self.system) is EmbeddedLowDNetwork:
#             latent = self.system.low_d_system.result
#             latent = self.trials_repeat(latent)
#         else:
#             latent = None
#         if inputs is not None:
#             inputs = self.trials_repeat(inputs)
            
#         rates   = self.trials_repeat(rates)
        unitspikesCount, trialVersion, trialType = self.load_spike_data()
        unitspikesCount = unitspikesCount[:]
        trialVersion = trialVersion[:]
        trialType = trialType[:]


        # myarray = np.ndarray((2296,202,90),buffer=data)
        unitspikesCount = unitspikesCount.transpose((2,0,1))
        self.num_trials = unitspikesCount.shape[0]
        num_steps_before_downsample = unitspikesCount.shape[1]
        self.net_size = unitspikesCount.shape[2]
        print(self.dt_cal)
        self.num_steps = int(num_steps_before_downsample*(1/(1000*self.dt_cal)))#
        
        
        unitspikesCount = np.reshape(unitspikesCount,(self.num_trials,1,num_steps_before_downsample,self.net_size))
        print(self.num_steps)
        unitspikesCount_ds = self.resample_data(unitspikesCount,num=self.num_steps)
        #sig.resample(unitspikesCount,self.num_steps,axis=2)
        spikes = unitspikesCount_ds
#         spikes  = self.spikify(rates, self.calcium_dynamics.dt)
        
#         pdb.set_trace()
        calcium = self.calcium_dynamics.integrate(num_steps=self.num_steps, inputs=spikes.transpose(2, 0, 1, 3)).transpose(1, 2, 0, 3)
        fluor   = calcium + np.random.randn(*calcium.shape)*self.sigma
        print('spikes size = ',spikes.shape, 'calcium size = ', calcium.shape)
#         pdb.set_trace()
        cells, cell_loc = self.generate_cells(num_cells=self.net_size,
                                              frame_width=self.frame_width,
                                              frame_height=self.frame_height,
                                              cell_radius=self.cell_radius)
        
        data_dict = {}
        for data, data_name in zip((spikes, calcium, fluor), 
                                   ('spikes', 'calcium', 'fluor')):
            if data is not None:
                data_dict['train_%s'%data_name], data_dict['valid_%s'%data_name] = self.train_test_split(data)
        
        data_dict['cells'] = cells
        data_dict['cell_loc'] = cell_loc
        data_dict['dt'] = self.calcium_dynamics.dt
        
        return data_dict
        
    def trials_repeat(self, data):
        data = data[..., None] * np.ones(self.num_trials)
        return data.transpose(3, 1, 0, 2)
        
    def spikify(self, rates, dt):
        return np.random.poisson(rates*dt)
    
    def calcify(self, spikes):
        return self.calcium_dynamics.integrate(num_steps=num_steps, inputs=spikes)
        
    def generate_cells(self, num_cells, frame_width, frame_height, cell_radius):
        import skimage.draw as draw
        cell_loc = np.random.uniform(low=np.array([[0.0], [0.0]])* np.ones((1, num_cells)),
                                     high=np.array([[frame_width], [frame_height]]) * np.ones((1, num_cells)))
        A = np.zeros((num_cells, frame_width + 2*cell_radius, frame_height + 2*cell_radius))
        for ix in range(num_cells):
            r, c = cell_loc[:, ix]
            rr, cc = draw.circle(r, c, radius=cell_radius)
            A[ix, rr, cc] += 1
        return A[:, cell_radius:-cell_radius, cell_radius:-cell_radius], cell_loc
    
    def train_test_split(self, data):
        num_trials, num_inits, num_steps, num_cells = data.shape
        num_train = int(self.trainp * num_trials)
        train_data = data[:num_train].reshape(num_train*num_inits, num_steps, num_cells)
        valid_data = data[num_train:].reshape((num_trials - num_train)*num_inits, num_steps, num_cells)
        return train_data, valid_data
    
class SyntheticCalciumVideoDataset(torch.utils.data.Dataset):

    
    def __init__(self, traces, cells, device='cpu', num_workers= 1, tmpdir='/tmp/'):
        
        super(SyntheticCalciumVideoDataset, self).__init__()
            
        self.cells  = cells
        self.traces = traces
        
        self.device = device
        
        num_trials, num_steps, num_cells = self.traces.shape
        num_cells, height, width = self.cells.shape
        num_channels = 1
        
        self.tempfile = tempfile.TemporaryFile(suffix='.dat', dir='/tmp/')
        self.tensors = (np.memmap(self.tempfile, dtype='float32', mode='w+', shape=(num_trials, 1, num_steps, height, width)),)
        
        def generate_video(trace, mmap, ix):
            res_ = (trace[..., np.newaxis, np.newaxis] * self.cells).sum(axis=1)[np.newaxis, ...]
            mmap[0][ix] = res_
        
        from joblib import Parallel, delayed
        Parallel(n_jobs=num_workers)(delayed(generate_video)(trace, self.tensors, ix) for ix, trace in enumerate(self.traces))
        
        self.dtype = self[0][0].dtype
        
    def __getitem__(self, ix):
        return (torch.from_numpy(self.tensors[0][ix]).to(self.device), )
    
    def __len__(self):
        # return traces.__len__()
        return len(self.traces)
    
    def close(self):
        self.tempfile.close()
        del self.tensors
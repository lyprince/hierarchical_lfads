import numpy as np
import oasis
import pdb
import argparse
import os

from utils import read_data, write_data

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str)
parser.add_argument('-t', '--tau', default=0.1, type=float)
parser.add_argument('-s', '--scale', default=1.0, type=float)
parser.add_argument('-k', '--known', action='store_true', default=False)
parser.add_argument('-f', '--flatten', action='store_true', default=False)
parser.add_argument('-n', '--normalize', action='store_true', default=False)
parser.add_argument('-z', '--undo_train_test_split', action='store_true', default=False)

def main():
    args = parser.parse_args()
    data_name = os.path.basename(args.data_path).split('.')[0]
    dir_name = os.path.dirname(args.data_path)
    
    data_dict = read_data(args.data_path)
    dt = data_dict['dt']
    g = np.exp(-dt/args.tau)
    
    train_size, steps_size, state_size = data_dict['train_fluor'].shape
    valid_size, steps_size, state_size = data_dict['valid_fluor'].shape
    data_size = train_size + valid_size
    data = np.zeros((data_size, steps_size, state_size))
    
    if args.undo_train_test_split:
        train_idx = data_dict['train_idx']
        valid_idx = data_dict['valid_idx']
        data[train_idx] = data_dict['train_fluor']
        data[valid_idx] = data_dict['valid_fluor']
        
    else:
        data[:train_size] = data_dict['train_fluor']
        data[train_size:] = data_dict['valid_fluor']
    
    if args.flatten:
        data = data.reshape(data_size * steps_size, state_size).transpose()
    else:
        data = data.transpose(0, 2, 1)
        data = data.reshape(data_size * state_size, steps_size)
        data = np.hstack((np.zeros((data_size * state_size, 1)), data))

    if args.known:
        S, C = deconvolve_calcium_known(data, 
                                        g=g, 
                                        s_min=args.scale/2)
    else:
        if args.normalize:
            data = max_normalize(data.T, axis=0).T
        S, C, bias, G, gain, rval = deconvolve_calcium_unknown(data,
                                                               g=g,
                                                               snr_thresh=args.scale)
        tau = -dt/(np.log(G))
        
    if args.flatten:
        data = data.reshape(data_size, steps_size, state_size)
        S = S.reshape(data_size, steps_size, state_size)
        C = C.reshape(data_size, steps_size, state_size)
        
    else:
        data = data.reshape(data_size, state_size, steps_size+1).transpose(0, 2, 1)[:, 1:]
        S = S.reshape(data_size, state_size, steps_size+1).transpose(0, 2, 1)[:, 1:]
        C = C.reshape(data_size, state_size, steps_size+1).transpose(0, 2, 1)[:, 1:]
        
        if not args.known:
            bias = bias.reshape(data_size, state_size).mean(axis=0)
            tau  = tau.reshape(data_size, state_size).mean(axis=0)
            gain = gain.reshape(data_size, state_size).mean(axis=0)
        
    if args.undo_train_test_split:
        train_fluor = data[train_idx]
        valid_fluor = data[valid_idx]
        train_ospikes  = S[train_idx]
        valid_ospikes  = S[valid_idx]
        train_ocalcium = C[train_idx]
        valid_ocalcium = C[valid_idx]
        
    else:
        train_fluor = data[:train_size]
        valid_fluor = data[train_size:]
        train_ospikes  = S[:train_size]
        valid_ospikes  = S[train_size:]
        train_ocalcium = C[:train_size]
        valid_ocalcium = C[train_size:]
        
    data_dict['train_fluor'] = train_fluor
    data_dict['valid_fluor'] = valid_fluor
    
    data_dict['train_ospikes'] = train_ospikes
    data_dict['valid_ospikes'] = valid_ospikes
    
    data_dict['train_ocalcium'] = train_ocalcium
    data_dict['valid_ocalcium'] = valid_ocalcium
    
    if not args.known:
        data_dict['obs_gain_init'] = gain
        data_dict['obs_bias_init'] = bias
        data_dict['obs_tau_init'] = tau
        data_dict['obs_var_init'] = (gain/args.scale)**2
    
    arg_string =  '_o%s'%('k' if args.known else 'u')
    arg_string += '_t%s'%(str(args.tau))
    arg_string += '_s%s'%(str(args.scale))
    arg_string.replace('.', '-')
    
    write_data(os.path.join(dir_name, data_name) + arg_string, data_dict)

def deconvolve_calcium_known(X, g=0.9, s_min=0.5):
    S = np.zeros_like(X)
    C = np.zeros_like(X)
    for ix, x in enumerate(X):
            c,s = oasis.functions.oasisAR1(x, g=g, s_min=0.5)
            S[ix] = s.round()
            C[ix] = c
    return S, C

def deconvolve_calcium_unknown(X, g=0.9, snr_thresh=3):
    '''
    Deconvolve calcium traces to spikes
    '''
    
    S = np.zeros_like(X)
    C = np.zeros_like(X)
    
    B = []
    G = []
    L = []
    M = []
    R = []
    
    b_init = compute_mode(X)
    
    for ix, x in enumerate(X):
        c, s, b, g, lam = oasis.functions.deconvolve(x, b=b_init, g=[g], penalty=1, max_iter=5)
        sn = (x-c).std(ddof=1)
        c, s, b, g, lam = oasis.functions.deconvolve(x, b=b, penalty=1, g=[g], sn=sn, max_iter=5)
        sn = (x-c).std(ddof=1)
        c, s = oasis.oasis_methods.oasisAR1(x-b, g=g, lam=lam, s_min=sn*snr_thresh)
        r = np.corrcoef(c, x)[0, 1]
        
        S[ix] = np.round(s/(sn*snr_thresh))
        C[ix] = c
        
        B.append(b)
        G.append(g)
        L.append(lam)
        M.append(sn * snr_thresh)
        R.append(r)
            
    B = np.array(B)
    G = np.array(G) 
    L = np.array(L)
    M = np.array(M)
    R = np.array(R)
    
    return S, C, B, G, M, R

def max_normalize(X, axis=0):
    X = X - compute_mode(X)
    return X/X.max()

def compute_mode(X):
    h, b  = np.histogram(X.ravel(), bins='auto')
    xvals = (b[1:] + b[:-1])/2
    return xvals[h.argmax()]

if __name__ == '__main__':
    main()
    
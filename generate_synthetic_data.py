#!/usr/bin/env python

from synthetic_data import SyntheticCalciumDataGenerator, ShenoyCalciumDataGenerator
from utils import write_data
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--system', default='lorenz', type=str)
parser.add_argument('-o', '--output', default='./', type=str)
parser.add_argument('-s', '--seed', default=100, type=int)
parser.add_argument('-p', '--parameters', type=str)
parser.add_argument('--trials', default=10, type=int)
parser.add_argument('--inits', default=100, type=int)
parser.add_argument('--cells', default=50, type=int)
parser.add_argument('--steps', default=100, type=int)
parser.add_argument('--rate_scale', default=5.0, type=float)
parser.add_argument('--trainp', default=0.8, type=float)
parser.add_argument('--dt_spike', default=0.01, type=float)
parser.add_argument('--dt_sys', default=0.01, type=float)
parser.add_argument('--burn_steps', default=0, type=int)
parser.add_argument('--shenoy_dir', default='./', type=str)

def main():
    args = parser.parse_args()
    if args.parameters:
        params_dict = yaml.load(open(args.parameters), Loader=yaml.FullLoader)
        for key, val in params_dict.items():
            args.__setattr__(key, val)
            print('%s : %s'%(key, str(args.__getattribute__(key))), flush=True)
    
    if args.system == 'lorenz':
        from synthetic_data import LorenzSystem, EmbeddedLowDNetwork
        
        lorenz = LorenzSystem(num_inits= args.inits,
                              dt= args.dt_sys)
        
        net = EmbeddedLowDNetwork(low_d_system = lorenz,
                                  net_size = args.cells,
                                  base_rate = args.rate_scale,
                                  dt = args.dt_sys)
        
    elif args.system == 'chaotic-rnn':
        from synthetic_data import ChaoticNetwork, RandomPerturbation
        
        inputs = RandomPerturbation(t_span=[0.25, 0.75], scale=10)
        
        net = ChaoticNetwork(num_inits= args.inits,
                             max_rate= args.rate_scale,
                             net_size = args.cells,
                             weight_scale = 2.5,
                             dt=args.dt_sys,
                             inputs= inputs)
        
    # generate data
    if args.system == 'shenoy':
        generator = ShenoyCalciumDataGenerator(data_dir    = args.shenoy_dir,
                                               trainp      = args.trainp, 
                                               tau_cal     = 0.3, 
                                               dt_cal      = args.dt_spike,
                                               sigma       = 0.2)
        
    else:
        generator = SyntheticCalciumDataGenerator(system     = net,
                                                  seed       = args.seed,
                                                  trainp     = args.trainp,
                                                  burn_steps = args.burn_steps,
                                                  num_steps  = args.steps,
                                                  num_trials = args.trials,
                                                  tau_cal    = 0.3,
                                                  dt_cal     = args.dt_spike,
                                                  sigma      = 0.2)

    data_dict = generator.generate_dataset()
    
    # save
    
    print('Saving to %s/%s_%03d'%(args.output, args.system, args.seed), flush=True)
    write_data('%s/%s_%03d'%(args.output, args.system, args.seed), data_dict)
    
if __name__ == '__main__':
    main()
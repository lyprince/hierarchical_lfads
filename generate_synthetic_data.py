from synthetic_data import generate_lorenz_data, generate_chaotic_rnn_data
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
parser.add_argument('--steps_in_bin', default=1, type=int)
parser.add_argument('--rate_scale', default=5.0, type=float)
parser.add_argument('--trainp', default=0.8, type=float)
parser.add_argument('--dt_spike', default=0.01, type=float)
parser.add_argument('--dt_sys', default=0.01, type=float)

def main():
    args = parser.parse_args()
    if args.parameters:
        params_dict = yaml.load(open(args.parameters), Loader=yaml.FullLoader)
        for key, val in params_dict.items():
            args.__setattr__(key, val)
            print('%s : %s'%(key, str(args.__getattribute__(key))), flush=True)
    if args.system == 'lorenz':
        data_dict = generate_lorenz_data(N_cells          = args.cells,
                                         N_inits          = args.inits,
                                         N_trials         = args.trials,
                                         N_steps          = args.steps,
                                         N_stepsinbin     = args.steps_in_bin,
                                         dt_spike         = args.dt_spike,
                                         dt_lorenz        = args.dt_sys,
                                         base_firing_rate = args.rate_scale,
                                         save_dir         = args.output,
                                         seed             = args.seed,
                                         save             = True)
    elif args.system == 'chaotic-rnn':
        data_dict = generate_chaotic_rnn_data(Ncells   = args.cells,
                                              Ninits   = args.inits,
                                              Ntrial   = args.trials,
                                              Nsteps   = args.steps,
                                              dt_spike = args.dt_spike,
                                              dt_rnn   = args.dt_sys,
                                              maxRate  = args.rate_scale,
                                              save_dir = args.output,
                                              seed     = args.seed,
                                              save     = True)
    else:
        raise ValueError('Unrecognised simulator system argument. Must choose lorenz or chaotic-rnn')

if __name__ == '__main__':
    main()

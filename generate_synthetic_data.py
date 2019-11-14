from synthetic_data import generate_lorenz_data, generate_chaotic_rnn_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--system', default='lorenz', nargs=1, type='str')
parser.add_argument('-o', '--output', default='./', nargs=1, type='str')
parser.add_argument('-s', '--seed', default=100, nargs=1, type=int)
parser.add_argument('--trials', default=10, nargs=1, type=int)
parser.add_argument('--inits', default=100, nargs=1, type=int)
parser.add_argument('--cells', default=50, nargs=1, type=int)
parser.add_argument('--steps', default=100, nargs=1, type=int)
parser.add_argument('--steps_in_bin', default=1, nargs=1, type=int)
parser.add_argument('--rate_scale', default=5, nargs=1, type=int)
parser.add_argument('--trainp', default=0.8, nargs=1, type=float)
parser.add_argument('--dt_spike', default=0.01, nargs=1, type=float)
parser.add_argument('--dt_sys', default=0.01, nargs=1, type=float)

def main():
    if args.system == 'lorenz':
        data_dict = generate_lorenz_data(N_cells=args.cells,
                                         N_inits=args.inits,
                                         N_trials=args.trials,
                                         N_steps=args.steps,
                                         N_stepsinbin=args.steps_in_bin,
                                         dt_spike=args.dt_spike,
                                         dt_lorenz=args.dt_sys,
                                         base_firing_rate= args.rate_scale,
                                         save_dir=arg.output
                                         save=True)
    elif args.system == 'chaotic_rnn':
        data_dict = generate_chaotic_rnn_data(N_cells=args.cells,
                                              N_inits=args.inits,
                                              N_trials=args.trials,
                                              N_steps=args.steps,
                                              dt_spike=args.dt_spike,
                                              dt_rnn=dt_sys,
                                              maxRate= args.rate_scale,
                                              save_dir=arg.output
                                              save=True)
    else:
        raise ValueError('Unrecognised simulator system argument. Must choose lorenz or chaotic_rnn')

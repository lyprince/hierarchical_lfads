from synthetic_data import MLSpikeCalciumDataGenerator
from utils import read_data, write_data

def main():
    for ix in range(20):
        data_path = './synth_data/lorenz_%i'%((ix+1)*1000)
        data_dict = read_data(data_path)
        mls_gen = MLSpikeCalciumDataGenerator(train_calcium=data_dict['train_calcium'], valid_calcium=data_dict['valid_calcium'],
                                              n=2.0,
                                              A=1.0,
                                              gamma=0.01)

        train_fluor_mlspike, valid_fluor_mlspike = mls_gen.generate_dataset()
        data_dict['train_fluor'] = train_fluor_mlspike
        data_dict['valid_fluor'] = valid_fluor_mlspike
        write_data(data_path+'_mlspike', data_dict)
        
if __name__ == '__main__':
    main()
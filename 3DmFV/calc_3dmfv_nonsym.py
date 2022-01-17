# Calculating and pickleling the 3dmfv
import myutilsnonsym
import pickle

DATA_PATH = '../../0_data/hdf/'
SAVE_PATH = './data/'

for trn_size in [(3, 3, 9), (5, 5, 10), (8, 8, 12)]:
    for files in ['train', 'val', 'test']:
        # trn_size = (gmm_size, ) * 3
        trn_size_str = 'x'.join(map(str, trn_size))
        trees = myutils.TreeData(f'{DATA_PATH}{files}.h5',
                                 transform=trn_size,
                                 data_augmentation=False)
        with open(f'{SAVE_PATH}{files}_{trn_size_str}.pickle', 'wb') as sfile:
            pickle.dump(trees, sfile)

myutils.call_home('Done with 3DmFV calculation')

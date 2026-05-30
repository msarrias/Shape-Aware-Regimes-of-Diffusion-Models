import Diffusion

def load_config(DATASET):
    config = Diffusion.TrainingConfig()
    config.DATASET = DATASET
    
    config.DEVICE = 'cuda:0'
    config.LR = 1e-4
    config.N_STEPS = 150000 #int(5e5)+1
    config.path_save = '../Saves/'
    
    if DATASET == 'MNIST':
        config.IMG_SHAPE = (1, 32, 32)
        config.BATCH_SIZE = 128
        config.path_data = '../data/'
        config.CENTER = True
        config.STANDARDIZE = False
        config.n_images = 15000 #10000
        config.dataset_params = {"name": "MNIST", 
                                 "classes": [0, 1, 8],
                                 "props": [1/3, 1/3, 1/3]}
    else:
        raise Exception('Dataset {:s} not implemented'.format(DATASET))
        
    return config
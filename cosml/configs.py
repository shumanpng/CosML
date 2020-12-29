save_dir = './output/'
base_data_dir = '../../CrossDomainFewShot_original/filelists/'
datasets = ['miniImagenet', 'cars', 'cub', 'places', 'plantae']
splits_dir = '../data/crossdomain_data/'

options = {
            'pretrain': {
                'dataset': datasets,
                'model': 'Conv4-fixchannels-64',
                'method': 'pretrain',
                'train_type': 'nonepisodic',
                'name': 'feature_extractor',
                'splits_dir': splits_dir,
                'train_aug': True,
                'save_freq': 5

            },

            'meta_train': {
                    'dataset': datasets,
                    'model': 'Conv1',
                    'method': 'cosml',
                    'name' : 'metalearners',
                    'train_type': 'episodic',
                    'save_freq': 2,#10,
                    'start_epoch': 0,
                    'stop_epoch': 400,
                    'mixed_task_batch_size': 25,
                    'pure_task_batch_size': 25,
                    'mixed_val_batch_size': 5,
                    'pure_val_batch_size': 5
            },

            'meta_test': {
                    'num_classes': 861,
                    'dataset': datasets,
                    'model': 'Conv1',
                    'method': 'maml',
                    'name': 'metalearners',
                    'save_dir': save_dir,
                    'data_dir': base_data_dir,
                    'splits_dir': splits_dir
            }
}

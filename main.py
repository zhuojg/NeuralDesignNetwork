import os
import argparse
import configparser
import datetime

from models.pipeline import NeuralDesignNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--part', choices=['relation', 'generation', 'refinement', 'all'])
parser.add_argument('--checkpoint_path', default=None)
parser.add_argument('--output_dir', default=None)
args = parser.parse_args()

config_parser = configparser.ConfigParser()
config_parser.read('config.ini')
config = config_parser['NDN']

category_list = ['image', 'text', 'background', 'header', 'text over image', 'header over image']
pos_relation_list = ['surrounding', 'inside', 'left of', 'above', 'right of', 'below', 'unknown']
size_relation_list = ['bigger', 'smaller', 'same', 'unknown']


if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_config = {
        'learning_rate': config.getfloat('learning_rate'),
        'beta_1': config.getfloat('beta_1'),
        'beta_2': config.getfloat('beta_2'),
        'data_dir': config['data_dir'],
        'test_data_dir': config['test_data_dir'],
        'sample_data_dir': config['sample_data_dir'],
        'mask_rate': config.getfloat('mask_rate'),
    }

    if args.train:
        checkpoint_dir = os.path.join(config['checkpoint_dir'], current_time)
        sample_dir = os.path.join(config['sample_dir'], current_time)
        train_sample_dir = os.path.join(sample_dir, 'train')
        test_sample_dir = os.path.join(sample_dir, 'test')

        log_dir = os.path.join(config['log_dir'], current_time)
        
        if args.save:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            if not os.path.exists(sample_dir):
                os.makedirs(train_sample_dir)
                os.makedirs(test_sample_dir)
                
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        
        training_config = {
            'checkpoint_dir': checkpoint_dir,
            'log_dir': log_dir,
            'train_sample_dir': train_sample_dir,
            'batch_size': config.getint('batch_size'),
            'lambda_cls': config.getfloat('lambda_cls'),
            'lambda_recon': config.getfloat('lambda_recon'),
            'lambda_kl_1': config.getfloat('lambda_kl_1'),
            'lambda_kl_2': config.getfloat('lambda_kl_2'),
            'max_iteration_number': int(config.getfloat('max_iteration_number')),
            'checkpoint_every': int(config.getfloat('checkpoint_every')), 
            'checkpoint_max_to_keep': config.getint('checkpoint_max_to_keep'),
            'sample_every': int(config.getint('sample_every'))
        }

        training_config = {**training_config, **model_config}
        
        model = NeuralDesignNetwork(
            category_list=category_list, 
            pos_relation_list=pos_relation_list,
            size_relation_list=size_relation_list,
            config=training_config,
            save=args.save, 
            training=True)

        model.run(training_config)

    if args.test:
        # assert args.checkpoint_path and args.output_dir

        # if not os.path.exists(args.output_dir):
        #     os.makedirs(args.output_dir)

        assert args.checkpoint_path and args.output_dir
        
        model = NeuralDesignNetwork(
            category_list=category_list,
            pos_relation_list=pos_relation_list,
            size_relation_list=size_relation_list,
            config=model_config,
            save=args.save, 
            training=False
        )

        model.test(model_config, args.checkpoint_path, args.output_dir)

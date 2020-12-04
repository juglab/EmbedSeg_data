import copy
import os
from PIL import Image

import torch
from utils import transforms as my_transforms
import argparse
import json

# parser=argparse.ArgumentParser(description="")
# parser.add_argument("--exp_config")
# args_cluster=parser.parse_args()
# with open(args_cluster.exp_config) as f:
#     config = json.load(f)
#
# print("These are my config parameters:")
# for key in config.keys():
#     print("    config[{}] : {}".format(key, config[key]))
#
#DATA_DIR = config['DATA_DIR'] # TODO
DATA_DIR=os.environ.get('DATA_DIR') # TODO
# experiment_name = config['exp_name'] # TODO
# print("experiment_name : ", experiment_name)


args = dict(

    cuda=True,
    display=True, #TODO
    display_it=5,

    save=True,
    #save_dir='./exp/'+experiment_name,
    save_dir='./exp/dsb_nov10',
    resume_path=None,


    train_dataset = {
        #'name': 'cityscapes', #TODO
        'name': 'dsbreduced2018', #TODO
        #'name': 'ctc2017',
        #'name': 'usiigaci2019', #TODO
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'train',
            'size': 3000, #3000 --> for all, 1125 --> usiigaci
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomRotationsAndFlips',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'degrees': 90,
                        #'flip': 0,
                    }
                },
                {
                    'name': 'ToTensorFromNumpy',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 16, # 16--> all, 6 --> usiigaci
        'workers': 8 #TODO 8
    }, 

    val_dataset = {
        #'name': 'ctc2017',
        'name':'dsbreduced2018',
        #'name':'usiigaci2019',
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'val',
            'size': 1300, # mouse (8 fold) --> 1300,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomRotationsAndFlips',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'degrees': 90,
                        #'flip': 0,
                    }
                },
                {
                    'name': 'ToTensorFromNumpy',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 16, # 16 --> all
        'workers': 8 #TODO: 8
    }, 

    model = {
        'name': 'branched_erfnet', 
        'kwargs': {
            'num_classes': [4,1] #TODO: [3,1] earlier
        }
    }, 

    lr=5e-4,
    n_epochs=200, #TODO
    # loss options
    loss_opts={
        'to_center': 'com', # TODO com, median, learn
        'n_sigma': 2, #TODO, 1
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
)


def get_args():
    return copy.deepcopy(args)

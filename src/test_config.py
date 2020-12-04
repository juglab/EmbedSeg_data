import copy
import os
import torch
from utils import transforms as my_transforms

DATA_DIR=os.environ.get('DATA_DIR') #TODO
EXPERIMENT_NAME=os.environ.get('EXPERIMENT_NAME')
ap_thresh=float(os.environ.get('ap_thresh'))
#valStep= int(os.environ.get('valStep'))
seedThresh = float(os.environ.get('seedThresh'))

args = dict(
    ap_thresh=ap_thresh,
    minMaskSum=128,
    minUnclusteredSum=128,
    proposalSum=36,
    #valStep= valStep,
    seedThresh=seedThresh,
    n_sigma=2, #TODO
    tta=True,
    cuda=True,
    display=False,
    saveResults=True,  # TODO\
    saveImages=True,
    save_dir='./static/',
    checkpoint_path='./../../exp/' + EXPERIMENT_NAME + '/best_iou_model.pth', #TODO
    dataset= {
        'name': 'mousenuclei2020',
        'kwargs': {
            'root_dir': DATA_DIR,
            'type': 'test',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        }
    },
        
    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [4, 1], #TODO
        }
    }
)


def get_args():
    return copy.deepcopy(args)

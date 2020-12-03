"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import test_config
import torch
from datasets import get_dataset
from models import get_model
from utils.utils import Cluster, Visualizer
from utils.utils2 import matching_dataset

torch.backends.cudnn.benchmark = True
import numpy as np  # TODO
import tifffile

args = test_config.get_args()

n_sigma = args['n_sigma']
ap_thresh=args['ap_thresh']

minMaskSum=args['minMaskSum']
minUnclusteredSum=args['minUnclusteredSum']
proposalSum=args['proposalSum']

tta= args['tta']
valStep =args['valStep']
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

#if args['save']:
#    if not os.path.exists(args['save_dir']):
#        os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])

print("Val dataset is of length", len(dataset))

dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# load snapshot
if os.path.exists(args['checkpoint_path']):
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert (False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()

# cluster module
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigmay', 'sigmax', 'sigmaxy', 'seed'))  # TODO

def applyTTA(im, tta=False):
    if (tta):
        im1 = torch.flip(im, (2,))
        im2 = torch.flip(im, (3,))
        im3 = torch.flip(im1, (3,))

        output = model(im)
        output1 = model(im1)
        output2 = model(im2)
        output3 = model(im3)

        # detransform images
        if n_sigma==2:
            output1 = torch.flip(output1, (2,))
            output1[:, 1, ...] = -1 * output1[:, 1, ...]  # dx, dy, ...
            output2 = torch.flip(output2, (3,))
            output2[:, 0, ...] = -1 * output2[:, 0, ...]  # dx, dy, ...
            output3 = torch.flip(torch.flip(output3, (2,)), (3,))
            output3[:, 0:2, ...] = -1 * output3[:, 0:2, ...]
            outputConcatenated = torch.cat((output, output1, output2, output3), 0)
            output = torch.mean(outputConcatenated, 0, keepdim=True)
        elif n_sigma==3:
            output1 = torch.flip(output1, (2,))
            output1[:, 1, ...] = -1 * output1[:, 1, ...]  # dx, dy, ...
            output1[:, 4, ...] = -1 * output1[:, 4, ...]  # angle

            output2 = torch.flip(output2, (3,))
            output2[:, 0, ...] = -1 * output2[:, 0, ...]  # dx, dy, ...
            output2[:, 4, ...] = -1 * output2[:, 4, ...]  # angle

            output3 = torch.flip(torch.flip(output3, (2,)), (3,))
            output3[:, 0:2, ...] = -1 * output3[:, 0:2, ...]
            outputConcatenated = torch.cat((output, output1, output2, output3), 0)
            output = torch.mean(outputConcatenated, 0, keepdim=True)
    else:
        output = model(im)  # B 4+1 Y X
    return output

with torch.no_grad():

    #minMask = np.arange(64, 192, 32)  # 128
    #minUnclustered = np.arange(64, 192, 32)  # 128
    minMask =[128]
    minUnclustered=[128]
    minProposal = np.arange(18, 40, 2)  # 36
    minThresh=np.arange(0.5, 0.95, 0.05) # 0.05
    resultArray=np.zeros((len(dataset)//valStep + 1, len(minMask), len(minUnclustered), len(minProposal), len(minThresh)))

    for indi, i in enumerate(minMask):
        for indj, j in enumerate(minUnclustered):
            for indk, k in tqdm(enumerate(minProposal)):
                print("proposal:", k)
                for indl, l in enumerate(minThresh):
                    for indsample, sample in enumerate(dataset_it):
                        if(indsample%valStep==0):
                            im = sample['image']
                            instances = sample['instance'].squeeze()
                            #output = model(im)
                            output = applyTTA(im, tta)
                            instance_map, predictions = cluster.cluster(output[0], n_sigma=n_sigma, threshold=l,
                                                                            minMaskSum=i,
                                                                            minUnclusteredSum=j,
                                                                            proposalSum=k)  # TODO
                            sc = matching_dataset([instance_map.cpu().detach().numpy()], [instances.cpu().detach().numpy()],
                                                      thresh=ap_thresh, show_progress=False)
                            resultArray[indsample//valStep, indi, indj, indk, indl]=sc.accuracy


    print("result array shape", resultArray.shape)
    meanResult=np.mean(resultArray, axis=0)
    print(meanResult)
    bestIndMinMask, bestIndMinUnclustered, bestIndMinProposal, bestIndMinThresh = np.unravel_index(np.argmax(meanResult), meanResult.shape)
    print("Best accuracy is ", np.max(meanResult))
    print("Best metaparams are", minMask[bestIndMinMask], minUnclustered[bestIndMinUnclustered], minProposal[bestIndMinProposal], minThresh[bestIndMinThresh])

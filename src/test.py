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
import numpy as np #TODO
import tifffile
from tifffile import imsave
args = test_config.get_args()

n_sigma = args['n_sigma']
ap_thresh=args['ap_thresh']

minMaskSum=args['minMaskSum']
minUnclusteredSum=args['minUnclusteredSum']
proposalSum=args['proposalSum']

tta= args['tta']
seedThresh = args['seedThresh']

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

print("length of test data is", len(dataset_it))

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)


# load snapshot
if os.path.exists(args['checkpoint_path']):
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert(False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()

# cluster module
cluster = Cluster()

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
    resultList=[] #TODO
    imageFileNames=[]
    for sample in tqdm(dataset_it):

        im = sample['image'] # B 1 Y X
        im=im[..., :1000, :1000]
        output=applyTTA(im, tta)
        instances = sample['instance'].squeeze() #  Y X  (squeeze takes away first two dimensions)
        instance_map, predictions = cluster.cluster(output[0], n_sigma=n_sigma, threshold=seedThresh,
                                                    minMaskSum=minMaskSum, minUnclusteredSum=minUnclusteredSum, proposalSum=proposalSum) #TODO
        base, _ = os.path.splitext(os.path.basename(sample['im_name'][0])) #TODO
        imageFileNames.append(base)
        sc=matching_dataset([instance_map.cpu().detach().numpy()], [instances.cpu().detach().numpy()], thresh=ap_thresh) #TODO
        print("Accuracy: {:.03f}".format(sc.accuracy), flush=True)
        resultList.append(sc.accuracy) #TODO


        if args['saveImages'] and ap_thresh==0.5:
            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            instances_file = os.path.join(args['save_dir'], 'instances/', base + '.tif')  # TODO
            imsave(instances_file, instance_map.cpu().detach().numpy())
            gt_file =os.path.join(args['save_dir'], 'gt/', base + '.tif')  # TODO
            imsave(gt_file, instances.cpu().detach().numpy())

    # do for the complete set of images
    if args['saveResults']:
        txt_file = os.path.join(args['save_dir'], 'results/combined_AP' + str(ap_thresh) + '.txt')
        with open(txt_file, 'w') as f:
            f.writelines("ImageFileName, minMaskSum, minUnclusteredSum, proposalSum, seedThresh, Intersection Threshold, accuracy \n")
            f.writelines("+++++++++++++++++++++++++++++++++\n")
            for ind, im_name in enumerate(imageFileNames):
                im_name_png = im_name + '.png'
                score = resultList[ind]
                f.writelines(
                    "{} {:.02f} {:.02f} {:.02f} {:.02f} {:.02f} {:.03f} \n".format(im_name_png, minMaskSum, minUnclusteredSum, proposalSum, seedThresh, ap_thresh, score))
            f.writelines("+++++++++++++++++++++++++++++++++\n")
            f.writelines("Average Precision (AP)  {:.02f} {:.03f}\n".format(ap_thresh, np.mean(resultList)))

    print("mean result", np.mean(resultList))

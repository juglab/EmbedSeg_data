"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import math
import torch
from torch.autograd import Variable
from matplotlib.patches import Ellipse
import tifffile


class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)

class Visualizer:

    def __init__(self, keys):
        self.wins = {k:None for k in keys}

    def display(self, image, key, title, title2=None, com_x=None, com_y=None, center_learn_x=None, center_learn_y=None,
                samples_x=None, samples_y=None,
                sample_spatial_embedding_x=None, sample_spatial_embedding_y=None,
                sigma_x = None, sigma_y = None, angle=None,
                color_sample=None, color_embedding=None):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1
    
        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)
    
        fig, ax = self.wins[key]
        fig.frameon=False
        #fig.set_size_inches
        #fig.set_size_inches(w, h)
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1
    
        assert n_images == n_axes
    
        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            if (key=='image'):
                ax.imshow(self.prepare_img(image), cmap='gray')
                #from datetime import datetime
                #now = datetime.now()
                #current_time = now.strftime("%H:%M:%S")
                #tifffile.imsave('/home/manan/Desktop/images/' + key + '_' + current_time + '.tif',
                                #self.prepare_img(image))
            elif(key=='prediction' or key=='groundtruth'):
                #from datetime import datetime
                #now = datetime.now()
                #current_time = now.strftime("%H:%M:%S")
                #tifffile.imsave('/home/manan/Desktop/images/'+key+'_'+current_time+'.tif', self.prepare_img(image))
                ax.imshow(self.prepare_img(image))
            else:
                ax.imshow(self.prepare_img(image), cmap='gray')
            ax.set_title(title)
            if (key == 'center'):
                if (com_x is not None and com_y is not None and samples_x is not None and samples_y is not None):
                    # print("number of objects are", len(color_sample.items()))
                    for i in range(len(color_sample.items())):
                        ax.plot(com_x[i+1], com_y[i+1], color=color_embedding[i+1], marker='x')
                        ax.scatter(samples_x[i+1], samples_y[i+1], color = color_sample[i+1], marker='+')
                        ax.scatter(sample_spatial_embedding_x[i+1], sample_spatial_embedding_y[i+1], color=color_embedding[i+1], marker='.')
                        ellipse=Ellipse((com_x[i+1], com_y[i+1]), width=sigma_x[i+1], height=sigma_y[i+1], angle=angle[i+1]*180/math.pi, color=color_embedding[i+1], alpha=0.5)
                        ax.add_artist(ellipse)

                #from datetime import datetime
                #now = datetime.now()
                #current_time = now.strftime("%H:%M:%S")
                #ax.set_title(title)
                #fig.savefig('/home/manan/Desktop/images/'+key+'_'+current_time+'.png')

        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]))
                if i==0:
                    ax[i].set_title(title)
                elif i==1:
                    ax[i].set_title(title2)
        plt.draw()
        self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

class Cluster:

    def __init__(self, ):

        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()

    



    def cluster_with_gt(self, prediction, instance, n_sigma=1,):

        height, width = prediction.size(1), prediction.size(2)
    
        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
    
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w #TODO
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().cuda()
    
        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]
    
        for id in unique_instances:
    
            mask = instance.eq(id).view(1, height, width)
    
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1
    
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)

            s = torch.exp(s*10)  # n_sigma x 1 x 1 #TODO
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = (dist > 0.5)
            instance_map[proposal] = id
    
        return instance_map

    def cluster(self, prediction, n_sigma=3, threshold=0.5, minMaskSum=128, minUnclusteredSum=128, proposalSum=36):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w #TODO


        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2+n_sigma:2+n_sigma + 1])  # 1 x h x w
       
        instance_map =torch.zeros(height, width).short() #TODO
        instances = []

        count = 1
        mask = seed_map > 0.5

        if mask.sum() > minMaskSum:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda() #TODO
            instance_map_masked=torch.zeros(mask.sum()).short().cuda()

            
            while(unclustered.sum() > minUnclusteredSum):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < threshold:
                    break
                center = spatial_emb_masked[:, seed:seed+1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0))

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > proposalSum: #TODO: with 48, you get 0.85, with 42 you het 0.86, with 36 you get
                    if unclustered[proposal].sum().float()/proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask =torch.zeros(height, width).short() #TODO
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu() #TODO
                        instances.append(
                            {'mask': instance_mask.squeeze()*255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances

class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        count = 0
        for key in self.data:#TODO earlier self.data
            if(count<3):
                keys.append(key)
                data = self.data[key]
                ax.plot(range(len(data)), data, marker='.')
                count+=1
        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

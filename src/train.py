import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import train_config
from criterions.my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer
import torch.nn as nn


torch.backends.cudnn.benchmark = True
import matplotlib.cm as cm
import numpy
import seaborn as sns


def tanh(x, k):
    return (torch.exp(x / k) - torch.exp(-x / k)) / (torch.exp(x / k) + torch.exp(-x / k))

args = train_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])


train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)


# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])

val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)


# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

# set criterion
criterion = SpatialEmbLoss(**args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)

def lambda_(epoch):
    return pow((1-((epoch)/args['n_epochs'])), 0.9)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_,)

# clustering
cluster = Cluster()

# Visualizer
# visualizer = Visualizer(('image', 'pred', 'sigma', 'seed')) #TODO
visualizer = Visualizer(('image', 'pred', 'prediction', 'groundtruth', 'center'))

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# resume
start_epoch = 0
best_iou = 0
if args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    start_epoch = state['epoch'] + 1
    best_iou = state['best_iou']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']

def degrid(x, n, amax):
    return int(x*n/amax +1)

def addSamples(samples, ax,  n, amax):
    samples_list=[]
    for i in range(samples.shape[1]):
        samples_list.append(degrid(samples[ax, i], n, amax))
    return samples_list

from torch.autograd import Variable

def computeCovariance(s, nsigma):
    sigmaxy=0
    covariance=Variable(torch.zeros(2,2)).cuda()
    covariance[0,0]=0.5/s[0]
    covariance[1,1]=0.5/s[1]
    covariance[0,1]=sigmaxy
    covariance[1, 0]=sigmaxy
    return covariance


def getSigmaAngles(covariance, nsigma):
    if nsigma==2:
        sigma_x=covariance[0,0]
        sigma_y=covariance[1,1]
        angle=0
    elif nsigma==3:
        d, v=torch.eig(covariance, eigenvectors=True)
        # note that d
        if(d[0,0] <= d[1,0]):
            sigma_x=d[0, 0]
            sigma_y=d[1, 0]
            angle = torch.atan2(v[1, 0], v[0, 0]) # by defualt y followed by x
        elif(d[1,0] < d[0,0]):
            sigma_x = d[1, 0]
            sigma_y = d[0, 0]
            angle = torch.atan2(v[1, 1], v[0, 1])  # by defualt y followed by x
    return sigma_x, sigma_y, angle


def train(epoch):

    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):
        
        im = sample['image']

        instances = sample['instance'].squeeze()
        class_labels = sample['label'].squeeze()
        #centre_map=sample['centre_map'].squeeze()
        #print("Center map shape is", centre_map.shape)
        # ids = sample['ids']
        # centre_x=sample['centre_x']
        # centre_y=sample['centre_y']
        output = model(im) # B 5 Y X
        loss = criterion(output, instances, class_labels, **args['loss_w'])
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args['display'] and i % args['display_it'] == 0:
            with torch.no_grad():
                visualizer.display(im[0], 'image', 'image') #TODO
                
                predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])
                visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred', 'predictions', 'groundtruth') #TODO

                instance_ids = instances[0].unique()
                instance_ids = instance_ids[instance_ids != 0]

                xm = torch.linspace(0, 2, 2048).view(
                    1, 1, -1).expand(1, 1024, 2048)
                ym = torch.linspace(0, 1, 1024).view(
                    1, -1, 1).expand(1, 1024, 2048)
                xym = torch.cat((xm, ym), 0)
                height, width = predictions.size(0), predictions.size(1)

                xym_s = xym[:, 0:height, 0:width].contiguous()
                spatial_emb = torch.tanh(output[0, 0:2]).cpu() + xym_s
                sigma = output[0, 2:2+args['loss_opts']['n_sigma']] # 2/3 Y X
                # output 16 5 Y X
                # spatial emb 2 Y X
                # in mask Y X
                # instances 16 Y X
                #colors = cm.rainbow(numpy.linspace(0, 1, 20))
                color_sample = sns.color_palette("dark")
                color_embedding = sns.color_palette("bright")
                color_sample_dic={}
                color_embedding_dic={}



                samples_x={}
                samples_y={}
                sample_spatial_embedding_x={}
                sample_spatial_embedding_y={}
                center_x={}
                center_y={}
                sigma_x={}
                sigma_y={}
                angle={}


                for id in instance_ids:
                    in_mask = instances[0].eq(id)
                    xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1) # 2 N

                    # xyin 2 N
                    # samples
                    perm = torch.randperm(xy_in.size(1))

                    idx = perm[:5]
                    samples = xy_in[:, idx]

                    samples_x[id.item()] = addSamples(samples, 0,  2047, 2)
                    samples_y[id.item()] = addSamples(samples, 1,  1023, 1)

                    # embeddings
                    spatial_emb_in = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)
                    samples_spatial_embeddings = spatial_emb_in[:, idx]

                    sample_spatial_embedding_x[id.item()]= addSamples(samples_spatial_embeddings, 0, 2047, 2)
                    sample_spatial_embedding_y[id.item()] = addSamples(samples_spatial_embeddings, 1, 1023, 1)

                    # com
                    if args['loss_opts']['to_center'] == 'com':
                        center = xy_in.mean(1)  # 2 x 1 x 1
                    elif args['loss_opts']['to_center'] == 'median':
                        center_temp = torch.median(xy_in, dim=1).values  # 2 x 1 x 1
                        imin = torch.argmin((xy_in[0, :] - center_temp[0]) ** 2 + (xy_in[1, :] - center_temp[1]) ** 2)
                        center = xy_in[:, imin]
                    elif args['loss_opts']['to_center'] == 'learn':
                        center = spatial_emb_in.mean(1)  # 2 x 1 x 1


                    center_x[id.item()] = degrid(center[0], 2047, 2)
                    center_y[id.item()] = degrid(center[1], 1023, 1)

                    # sigma
                    s = sigma[in_mask.expand_as(sigma)].view(args['loss_opts']['n_sigma'], -1).mean(1)


                    if (args['loss_opts']['n_sigma'] == 2):
                        s = torch.exp(s * 10)  # TODO
                    elif (args['loss_opts']['n_sigma'] == 3):
                        s[0] = torch.exp(s[0] * 10)  # TODO 1/2*sigmay^2
                        s[1] = torch.exp(s[1] * 10)  # TODO 1/2*sigmax^2
                        s[2] = tanh(s[2], 10)  # TODO
                    covariance = computeCovariance(s, args['loss_opts']['n_sigma'])
                    sigma_x_tmp, sigma_y_tmp, angle_tmp=getSigmaAngles(covariance, args['loss_opts']['n_sigma'])
                    sigma_x[id.item()] = degrid(torch.sqrt(sigma_x_tmp), 2047, 2)
                    sigma_y[id.item()] = degrid(torch.sqrt(sigma_y_tmp), 1023, 1)
                    angle[id.item()] = angle_tmp
                    # colors
                    color_sample_dic[id.item()] = color_sample[int(id%10)]
                    color_embedding_dic[id.item()] = color_embedding[int(id%10)]

                visualizer.display(predictions.cpu(), 'prediction', 'prediction', color_sample=color_sample_dic,
                                   color_embedding=color_embedding_dic)  # TODO
                visualizer.display(instances[0].cpu(), 'groundtruth', 'groundtruth', color_sample=color_sample_dic,
                                   color_embedding=color_embedding_dic)
                visualizer.display(instances[0]>0, 'center', 'center', com_x=center_x, com_y=center_y, center_learn_x=None,
                                   center_learn_y=None,
                                   samples_x=samples_x, samples_y=samples_y,
                                   sample_spatial_embedding_x=sample_spatial_embedding_x,
                                   sample_spatial_embedding_y=sample_spatial_embedding_y,
                                   sigma_x =sigma_x, sigma_y = sigma_y, angle =angle,
                                   color_sample= color_sample_dic, color_embedding = color_embedding_dic)


        loss_meter.update(loss.item())

    return loss_meter.avg

def val(epoch):

    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    # put model into eval mode
    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze()
            class_labels = sample['label'].squeeze()

            output = model(im)
            loss = criterion(output, instances, class_labels, **args['loss_w'], iou=True, iou_meter=iou_meter)
            loss = loss.mean()

            if args['display'] and i % args['display_it'] == 0:
                with torch.no_grad():
                    visualizer.display(im[0], 'image', 'image') #TODO
                
                    predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])
                    visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred', 'pred') #TODO
    
            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg

def save_checkpoint(state, is_best, epoch, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if (epoch%10==0):
        file_name2 = os.path.join(args['save_dir'], str(epoch) + "_"+ name)
        torch.save(state, file_name2)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth'))

for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    train_loss = train(epoch)
    val_loss, val_iou = val(epoch)

    print('===> train loss: {:.2f}'.format(train_loss))
    print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.add('iou', val_iou)
    logger.plot(save=args['save'], save_dir=args['save_dir'])
    
    is_best = val_iou > best_iou
    best_iou = max(val_iou, best_iou)
        
    if args['save']:
        state = {
            'epoch': epoch,
            'best_iou': best_iou, 
            'model_state_dict': model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data,

        }
        save_checkpoint(state, is_best, epoch)

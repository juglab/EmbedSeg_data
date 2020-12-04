
import glob
import os
from multiprocessing import Pool
import tifffile
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def process(tup):
    im, inst = tup
    
    #image_path = os.path.splitext(os.path.relpath(im, os.path.join(IMAGE_DIR, 'train')))[0] #TODO
    #image_path = os.path.join(IMAGE_DIR, 'crops', image_path)# TODO
    #image_path=os.path.join(CITYSCAPES_DIR, 'crops', image_path)
    #image_path=os.path.join(CITYSCAPES_DIR, 'crops', 'trainVal/images/',image_path.split('/')[2].split('set')[1]) #TODO
    image_path='/home/manan/Data/Fluo-C2DL-MSC/ctc2017/crops/trainVal/images/'
    #instance_path = os.path.splitext(os.path.relpath(inst, os.path.join(INSTANCE_DIR, 'train')))[0] #TODO
    #instance_path = os.path.join(INSTANCE_DIR, 'crops', instance_path)
    #instance_path = os.path.join(CITYSCAPES_DIR, 'crops', 'trainVal/masks', instance_path.split('/')[2].split('set')[1]) #TODO
    instance_path='/home/manan/Data/Fluo-C2DL-MSC/ctc2017/crops/trainVal/masks/'

    try:  # can't use 'exists' because of threads
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
    except FileExistsError:
        pass

    #image = Image.open(im) #TODO
    instance = np.asarray(Image.open(inst)).astype(np.int16) #TODO
    image=tifffile.imread(im) #TODO
    #instance=tifffile.imread(inst) #TODO
    #instance=plt.imread(inst).astype(np.uint16) #TODO
    #w, h = image.size #TODO
    h, w = image.shape
    instance_np = np.array(instance, copy=False)
    object_mask = np.logical_and(instance_np >= OBJ_ID * 1000, instance_np < (OBJ_ID + 1) * 1000)
    
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids!= 0]

    # loop over instances
    for j, id in enumerate(ids):
        
        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)
        
        jj = int(np.clip(ym-CROP_SIZE/2, 0, h-CROP_SIZE))
        ii = int(np.clip(xm-CROP_SIZE/2, 0, w-CROP_SIZE))
        
        if(image[jj:jj+CROP_SIZE, ii:ii+CROP_SIZE].shape==(CROP_SIZE, CROP_SIZE)):
            im_crop=image[jj:jj+CROP_SIZE, ii:ii+CROP_SIZE]
            instance_crop=instance[jj:jj+CROP_SIZE, ii:ii+CROP_SIZE]
            tifffile.imsave(image_path+os.path.basename(im)[:-4]+"_{:03d}.tif".format(j), im_crop) #TODO
            tifffile.imsave(instance_path+os.path.basename(im)[:-4]+"_{:03d}.tif".format(j), instance_crop) #TODO

if __name__ == '__main__':
    
    DATA_DIR=os.environ.get('DATA_DIR')

    IMAGE_DIR=os.path.join(DATA_DIR, 'dsb2018')##TODO
    #INSTANCE_DIR=os.path.join(DATA_DIR, 'dsb2018')##TODO
    INSTANCE_DIR=os.path.join(DATA_DIR, 'ctc2017/trainVal/*/masks/')
    OBJ_ID = 0
    CROP_SIZE=256 #TODO: 512

    # load images/instances
    images = glob.glob(os.path.join(IMAGE_DIR, '*.tif')) # TODO
    images.sort()
    

    instances = glob.glob(os.path.join(INSTANCE_DIR, '*.tif')) #TODO 
    instances.sort()
    

    with Pool(8) as p:
        r = list(tqdm(p.imap(process, zip(images,instances)), total=len(images)))

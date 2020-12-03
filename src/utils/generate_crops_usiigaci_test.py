"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
from multiprocessing import Pool
import tifffile
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import find_objects

def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x



def process(tup):
    im, inst = tup
    image_path='/media/manan/Samsung_T5/Manan/Data/ISBI/Usiigaci_original/crops-1/test/images/' #TODO change this to crops-0
    instance_path='/media/manan/Samsung_T5/Manan/Data/ISBI/Usiigaci_original/crops-1/test/masks/' #TODO

    try:  # can't use 'exists' because of threads
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
    except FileExistsError:
        pass

    instance = tifffile.imread(inst).astype(np.uint16)
    image = tifffile.imread(im)  # TODO

    image = normalize(image, 1, 99.8, axis=(0, 1))  # TODO for crops 0, comment this
    instance = fill_label_holes(instance)

    tifffile.imsave(image_path + os.path.basename(im), image)
    tifffile.imsave(instance_path + os.path.basename(inst), instance)
if __name__ == '__main__':
    
    DATA_DIR=os.environ.get('DATA_DIR')

    IMAGE_DIR=os.path.join(DATA_DIR, 'test/images/')##TODO
    INSTANCE_DIR=os.path.join(DATA_DIR, 'test/masks/')##TODO
    print(IMAGE_DIR)
    OBJ_ID = 0
    CROP_SIZE=256 #TODO: 512

    # load images/instances
    images = glob.glob(os.path.join(IMAGE_DIR, '*.tif')) # TODO
    print(len(images))
    images.sort()
    

    instances = glob.glob(os.path.join(INSTANCE_DIR, '*.tif')) #TODO 
    instances.sort()
    

    with Pool(8) as p:
        r = list(tqdm(p.imap(process, zip(images,instances)), total=len(images)))

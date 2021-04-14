import random

import numpy as np
import skimage.color as sc

import torch

def get_patch(*args, patch_size=96, scale=2, asymmetric = False):
    # lr image shape
    ih, iw = args[0].shape[:2]
    # lr image patch size
    ip = patch_size // scale
    # choose random coordinates from lr image w.r.t. patch size
    ix = random.randrange(0, iw - ip)
    iy = random.randrange(0, ih - ip)
    if asymmetric == False:
        # corresponding hr coordinates
        tx, ty = scale * ix, scale * iy
        # return the lr and hr patches
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            args[1][ty:ty + patch_size, tx:tx + patch_size, :]
        ]
    else:
        # corresponding hr coordinates
        tx, ty = ix, scale * iy
        # return the lr and hr patches
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            args[1][ty:ty + patch_size, tx:tx + ip, :]
        ]
    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]


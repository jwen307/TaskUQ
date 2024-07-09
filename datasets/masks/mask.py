#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask.py
    - Helper function to load the masks and to apply the mask to the data
"""

import numpy as np
import scipy.io 
import torch
import sys
import os

sys.path.append('../../')
from util import helper

def get_mask(accel=4, size=320, mask_type='knee'):

    #Get the root of the project
    root = helper.get_root()

    #Mask location
    mask_loc = os.path.join(root, 'datasets/masks/mask_accel{0}_size{1}_gro_{2}.mat'.format(accel, size, mask_type))
    
    #Load the matrix
    #mask = scipy.io.loadmat('../datasets/masks/mask_accel{0}_size{1}_gro_{2}.mat'.format(accel, size, mask_type))
    mask = scipy.io.loadmat(mask_loc)
    mask = mask['samp'].astype('float32')
    mask = torch.from_numpy(mask).unsqueeze(0)
    
    
    return mask


def get_random_mask(accel=4, img_size=320, mask_type='knee', acs_size=13):
    # Get the number of k-space lines to keep
    num_keep =  int(img_size/accel) - acs_size

    # Get a random mask for each sample
    #rand_idx = torch.randperm(img_size)[0:num_keep].numpy()
    rand_idx = torch.randperm(img_size)

    # Get the center values
    center_idx = torch.arange(acs_size) + int(img_size/2) - int(acs_size/2)

    # Remove the center values and get the remaining values
    rand_idx = rand_idx[~torch.isin(rand_idx, center_idx)]
    rand_idx = rand_idx[0:num_keep].numpy()

    # Create the mask
    mask = torch.zeros(img_size)
    mask[center_idx] = 1
    mask[rand_idx] = 1

    return mask.unsqueeze(-1).unsqueeze(0)

def apply_mask(data, mask):
    
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data


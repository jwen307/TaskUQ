#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

network_utilsl.py
    - Helper functions for data consistency, applying masks, etc.
"""
import torch
import numpy as np
import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np

import sigpy as sp
import sigpy.mri as mr

# Function to get the number of parameters in a model
def print_num_params(model):
    num_params = sum([np.prod(p.shape) for p in model.parameters()])
    print("Number of parameters: {:,}".format(num_params))



def coils_to_chans(multicoil_imgs):
    '''
    Convert image so there's 2*num_coil channels
    (batch, num_coils, img_size, img_size, 2) -> (batch, num_coils*2, img_size, img_size)
    '''
    # Add a batch dimension if needed
    if len(multicoil_imgs.shape) < 5:
        multicoil_imgs = multicoil_imgs.unsqueeze(0)

    b, c, h, w, _ = multicoil_imgs.shape

    multicoil_imgs = multicoil_imgs.permute(0, 1, 4, 2, 3).reshape(-1, c * 2, h, w)

    return multicoil_imgs


def chans_to_coils(multicoil_imgs):
    '''
    Convert image so there are two channels for real and imaginary and c coils
    '''

    # Add a batch dimension if needed
    if len(multicoil_imgs.shape) < 4:
        multicoil_imgs = multicoil_imgs.unsqueeze(0)

    b, c, h, w = multicoil_imgs.shape

    # Add a new dimension for real and imaginary
    multicoil_imgs = torch.stack([multicoil_imgs[:, 0:int(c):2, :, :], multicoil_imgs[:, 1:int(c):2, :, :]], dim=-1)

    return multicoil_imgs

def format_multicoil(multicoil_imgs, chans):
    '''
    Function to format the multicoil images to either coils or channels
    multicoil_imgs: Multicoil images (batch, num_coils, img_size, img_size, 2) or (batch, num_coils*2, img_size, img_size)
    '''

    # Add a batch dimension if needed
    if len(multicoil_imgs.shape) < 4:
        multicoil_imgs = multicoil_imgs.unsqueeze(0)

    # Turn into a tensor if needed
    if not torch.is_tensor(multicoil_imgs):
        multicoil_imgs = torch.tensor(multicoil_imgs)

    # Convert to channels if needed
    if chans:
        if multicoil_imgs.shape[-1] == 2:
            multicoil_imgs = coils_to_chans(multicoil_imgs)
        return multicoil_imgs
    # Convert to coils if needed
    else:
        if multicoil_imgs.shape[-1] != 2:
            multicoil_imgs = chans_to_coils(multicoil_imgs)
        return multicoil_imgs


def format_singlecoil(singlecoil_imgs, chans):
    '''
    Function to format the singlecoil images to either coils or channels
    singlecoil_imgs: Singlecoil images (batch, img_size, img_size, 2) or (batch, 2, img_size, img_size)
    '''

    # Add a batch dimension if needed
    if len(singlecoil_imgs.shape) < 4:
        singlecoil_imgs = singlecoil_imgs.unsqueeze(0)

    # Turn into a tensor if needed
    if not torch.is_tensor(singlecoil_imgs):
        singlecoil_imgs = torch.tensor(singlecoil_imgs)

    # Convert to channels if needed
    if chans:
        if singlecoil_imgs.shape[-1] == 2:
            singlecoil_imgs = singlecoil_imgs.permute(0,3,1,2)
        return singlecoil_imgs
    # Convert to coils if needed
    else:
        if singlecoil_imgs.shape[-1] != 2:
            singlecoil_imgs = singlecoil_imgs.permute(0,2,3,1)
        return singlecoil_imgs


def check_type(x, dtype):
    '''
    Function to check if x is of type dtype (if not, convert it)
    Check for to see if its array or tensor
    '''
    if dtype == 'tensor':
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
    elif dtype == 'array':
        if not isinstance(x, np.ndarray):
            x = x.numpy()
    return x
    


# Function to get the data consistent reconstruction
def get_dc_image(pred, cond, mask, norm_val=1):
    '''
    Function to get the data consistent reconstruction of the predicted image
    pred: Predicted image
    cond: Condition (zero-filled image)
    mask: Mask
    mean: Mean of the dataset (default 0)
    std: Normalizing factor of the dataset (default 1)

    return: Data consisten reconstruction (batch, coils, img_size, img_size, 2)
    '''

    # Make sure all inputs are correctly formatted
    norm_val = check_type(norm_val, 'tensor').to(cond.device)
    pred = check_type(pred, 'tensor').to(cond.device)
    mask = check_type(mask, 'tensor').to(cond.device)
    
    #Go from 16 channels to 8 coils and 2 complex dimensions
    pred = format_multicoil(pred, chans=False)
    cond = format_multicoil(cond, chans=False)

    # Get the mask formatted correctly. Should be (batch_size, 1, 320, 1)
    if len(mask.shape)<4:
        mask = mask.unsqueeze(0)
    
    #Get the zero-filled kspace
    zfkspace = fastmri.fft2c(unnormalize(cond, norm_val))
    
    #Get the kspace of the predicted image
    pred_kspace = fastmri.fft2c(unnormalize(pred, norm_val))

    # Get the masked k-spaces
    # mask2D = mask[0].permute(0,2,1).repeat(1,mask.shape[2],1)
    # mask2Dinv = mask2D*-1 + 1
    # masked_pred_k = mask2Dinv.unsqueeze(-1).unsqueeze(0).repeat(1,1,1,1,2) * pred_kspace
    # masked_zfkspace = mask2D.unsqueeze(-1).repeat(1,1,1,2) * zfkspace
    mask2D = mask.permute(0, 1,3,2).repeat(1, 1, mask.shape[2], 1)
    mask2Dinv = mask2D * -1 + 1
    masked_pred_k = mask2Dinv.unsqueeze(-1).repeat(1, 1, 1, 1, 2) * pred_kspace
    masked_zfkspace = mask2D.unsqueeze(-1).repeat(1, 1, 1, 1, 2) * zfkspace

    # Get the data consistent reconstruction
    dc_pred_k = masked_pred_k + masked_zfkspace
    
    # Normalize again
    dc_pred = fastmri.ifft2c(dc_pred_k)
    dc_pred = normalize(dc_pred, norm_val)

    # Put back in channel format
    dc_pred = format_multicoil(dc_pred, chans=True)
    
    #dc_pred is normalized again
    return dc_pred, dc_pred_k



def apply_mask(x, mask, norm_val=1, inv_mask = False, give_kspace=False):
    '''
    Function to apply the mask to the image
    std: normalizing value
    inv_mask: If true, invert the mask
    '''

    norm_val = check_type(norm_val, 'tensor').to(x.device)
    mask = check_type(mask, 'tensor').to(x.device)
    x = format_multicoil(x, chans=False)
    
    #Make the mask 2D
    if len(mask.shape)<4:
        mask = mask.unsqueeze(0)
    #mask2D = mask[0].permute(0,2,1).repeat(1,mask.shape[2],1)
    mask2D = mask.permute(0, 1, 3, 2).repeat(1, 1, mask.shape[2], 1)

    #Invert the mask if needed
    if inv_mask:
        mask2D = mask2D*-1 + 1

    #Get the kspace
    kspace = fastmri.fft2c(unnormalize(x, norm_val))

    #Apply the mask
    #masked_k = mask2D.unsqueeze(-1).unsqueeze(0).repeat(1,1,1,1,2) * kspace
    masked_k = mask2D.unsqueeze(-1).repeat(1, 1, 1, 1, 2) * kspace

    if give_kspace:
        # Renormalized
        masked_k = normalize(masked_k, norm_val)
        return masked_k

    #Not normalized
    masked_img = fastmri.ifft2c(masked_k)
    #Renormalized
    masked_img = normalize(masked_img, norm_val)

    #Get back into (batch, num_coils*2, img_size, img_size)
    masked_img = format_multicoil(masked_img, chans=True)

    return masked_img

    

#Convert multicoil to singlecoil
def multicoil2single(multicoil_imgs, maps=None, rss=False, norm_val=None):
    ''' 
    multicoil_imgs: (b, 16, h, w) or (b, 8, h, w, 2)
    maps: sensitivity maps
    rss: if true, do RSS reconstruction, else do coil-combined reconstruction
    '''

    if norm_val is not None:
        multicoil_imgs = unnormalize(multicoil_imgs, norm_val)
    
    #Get the coil images to be complex
    multicoil_imgs = format_multicoil(multicoil_imgs, chans=False)
        
    b, num_coils, h, w, _ = multicoil_imgs.shape

    # Do RSS reconstruction
    if rss:
        combo_imgs = fastmri.rss_complex(multicoil_imgs, dim=1).unsqueeze(1)

    # Do coil combined reconstruction
    else:
        maps = check_type(maps, 'tensor')
        if len(maps.shape) < 4:
            maps = maps.unsqueeze(0).cpu().numpy()
        maps = check_type(maps, 'array')

        combo_imgs = []

        # Coil-combine each image separately
        for i in range(b):
            with sp.Device(0):
                #Show coil combined estimate (not SENSE)
                S = sp.linop.Multiply((h,w), maps[i])
                #print(multicoil_imgs[i].shape)
                combo_img = S.H * tensor_to_complex_np(multicoil_imgs[i].cpu())

                combo_imgs.append(to_tensor(combo_img))

        combo_imgs = torch.stack(combo_imgs).unsqueeze(1)

    if norm_val is not None:
        combo_imgs = normalize(combo_imgs, norm_val)
        
    return combo_imgs


# Convert singlecoil to multicoil
def single2multicoil(singlecoil_imgs, maps=None, norm_val=None):
    '''
    multicoil_imgs: (b, 16, h, w) or (b, 8, h, w, 2)
    maps: sensitivity maps
    rss: if true, do RSS reconstruction, else do coil-combined reconstruction
    '''

    if norm_val is not None:
        singlecoil_imgs = unnormalize(singlecoil_imgs, norm_val)

    # Get the coil images to be complex
    singlecoil_imgs = format_multicoil(singlecoil_imgs, chans=False)

    b, num_coils, h, w, _ = singlecoil_imgs.shape


    # Do coil combined reconstruction
    maps = check_type(maps, 'tensor')
    if len(maps.shape) < 4:
        maps = maps.unsqueeze(0).cpu().numpy()
    maps = check_type(maps, 'array')

    combo_imgs = []

    # Coil-combine each image separately
    for i in range(b):
        with sp.Device(0):
            # Show coil combined estimate (not SENSE)
            S = sp.linop.Multiply((h, w), maps[i])
            combo_img = S * tensor_to_complex_np(singlecoil_imgs[i][0].cpu())

            combo_imgs.append(to_tensor(combo_img))

    combo_imgs = torch.stack(combo_imgs)

    if norm_val is not None:
        combo_imgs = normalize(combo_imgs, norm_val)

    return combo_imgs

# Get the magnitude images
def get_magnitude(x, maps=None, rss=False):
    '''
    Function to get the magnitude image from multicoil, singlecoil, complex or magnitude images
    x: input images
        multicoil: (b, 2*c, h, w) or (b, c, h, w, 2)
        singlecoil: (b, h, w, 2) or (b, h, w)

    output: magnitude images (b, 1, h, w)
    '''

    x = check_type(x, 'tensor')

    # Determine if the input is multicoil or singlecoil
    # Get the image dimensions first
    shape = list(x.shape)
    shape.sort()
    h,w = shape[-2:]

    # Multicoil case
    if np.prod(list(x.shape)[1:]) > h*w*2:
        # Convert to singlecoil
        x = multicoil2single(x, maps=maps, rss=rss)

        if rss:
            return x
        else:
            x = fastmri.complex_abs(x)
            return x

    # Already magnitude image
    elif np.prod(list(x.shape)[1:]) == h*w:
        return x

    # Singlecoil case
    else:
        if len(x.shape) < 4:
            x = x.unsqueeze(1)

        # Check if it is complex
        if x.shape[-1] == 2:
            x = fastmri.complex_abs(x)
        elif x.shape[1] == 2:
            x = fastmri.complex_abs(x.permute(0,2,3,1))

        return x.unsqueeze(1)

#Find the sensitivity maps
def get_maps(zf_imgs, num_acs, normalizing_val=None):
    '''
    Function to get the sensitivity maps from the zero-filled images
    zf_imgs: zero-filled images (b, 16, h, w) or (b, 8, h, w, 2)
    '''

    # Put in coil format
    zf_imgs = format_multicoil(zf_imgs, chans=False)
    b, num_coils, h, w, _ = zf_imgs.shape
        
    
    #Need to unnormalize zf_imgs before going to kspace
    if normalizing_val is not None:
        zf_imgs = unnormalize(zf_imgs, normalizing_val)
    
    #Get the kspace
    masked_kspace = fastmri.fft2c(zf_imgs)
    
    all_maps = []
    
    #Get the map for each sample in the batch
    for i in range(b):
        maps = mr.app.EspiritCalib(tensor_to_complex_np(masked_kspace[i].cpu()), 
                                    calib_width=num_acs,
                                    show_pbar=False, 
                                    crop=0.70, 
                                    device=sp.Device(0),
                                    kernel_width=6).run().get()
        
        all_maps.append(maps)
        
    all_maps = np.stack(all_maps)
        
    return all_maps

#Unnormalize the values
def unnormalize(imgs, normalizing_val):
    #Images should be (b, c, h, w) or (b, coils, h, w, 2)
    if len(imgs.shape) == 4:
        imgs = imgs * normalizing_val.reshape(-1,1,1,1).to(imgs.device)
        
    if len(imgs.shape) == 5:
        imgs = imgs * normalizing_val.reshape(-1,1,1,1,1).to(imgs.device)
        
    return imgs

#Normalize the values
def normalize(imgs, normalizing_val):
    #Images should be (b, c, h, w) or (b, coils, h, w, 2)
    if len(imgs.shape) == 4:
        imgs = imgs / normalizing_val.reshape(-1,1,1,1).to(imgs.device)

    if len(imgs.shape) == 5:
        imgs = imgs / normalizing_val.reshape(-1,1,1,1,1).to(imgs.device)

    return imgs


def get_nullspace(x, masks):
    '''
    Function to get the nullspace of the subsampling matrix
    x: image
    '''

    # Put in coil format
    x = format_multicoil(x, chans=False)
    masks = check_type(masks, 'tensor')

    # Go to kspace
    k = fastmri.fft2c(x)

    # Invert the mask
    if len(masks.shape) == 4:
        masks = masks[0]
    inv_mask = masks[0, :, 0].cpu() * -1 + 1

    # Get the A matrix for subsampling
    A = torch.eye(masks.shape[-2]).to(x.device)
    A = A[:, inv_mask > 0]

    # Get the subsampled kspace
    ksub = torch.matmul(coils_to_chans(k), A)

    # Return the nullspace
    return ksub

def get_zero_filled(ksub, masks):
    '''
    Funnction to get the zero-filled image from the nullspace components
    ksub: nullspace components
    '''

    # Invert the mask
    if len(masks.shape) == 4:
        masks = masks[0]
    inv_mask = masks[0, :, 0].cpu() * -1 + 1

    # Get the A matrix for subsampling
    A = torch.eye(masks.shape[-2]).cuda()
    A = A[:, inv_mask > 0]

    # Get the image
    k = torch.matmul(ksub, A.T)
    x = fastmri.ifft2c(chans_to_coils(k))
    x = coils_to_chans(x)

    # Return the nullspace
    return x


def fft_cols(data, norm='ortho'):
    # 1D FFT applied to the columns
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = fastmri.ifftshift(data, dim=[-3])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2,), norm=norm
        )
    )
    data = fastmri.fftshift(data, dim=[-3])

    return data

def ifft_cols(data, norm='ortho'):
    # 1D FFT columns
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = fastmri.ifftshift(data, dim=[-3])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2,), norm=norm
        )
    )
    data = fastmri.fftshift(data, dim=[-3])

    return data


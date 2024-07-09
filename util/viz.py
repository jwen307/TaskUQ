#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

viz.py
    - Functions for visualizing the images
"""


import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import fastmri
import sigpy as sp
import sigpy.mri as mr
from matplotlib.patches import Rectangle
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
import torchvision
import os
import torchvision.transforms.functional as func

import sys
sys.path.append("..")
from evals import fid, metrics
from util import network_utils as nu


def show_img(imgs, nrow=5, maps = None, rss = False, return_grid=False, colormap=None, scale_each=True, normalize=True, **kwargs):
    '''
    Show any type of image (real or complex) (multicoil or singlecoil)
    '''

    # Check if it is a tensor, if not make it a tensor
    imgs = nu.check_type(imgs, 'tensor').detach().cpu()

    # Get the magnitude images
    imgs = nu.get_magnitude(imgs,maps = maps, rss = rss)

    # Put the images in a grid and show them
    if 'val_range' in kwargs:
        grid = torchvision.utils.make_grid(imgs, nrow=int(nrow), normalize=normalize, value_range=kwargs['val_range'])

    else:
        grid = torchvision.utils.make_grid(imgs, nrow=int(nrow), scale_each=scale_each, normalize=normalize)


    if not return_grid:
        f = plt.figure()
        f.set_figheight(15)
        f.set_figwidth(15)

        if 'rect' in kwargs:
            for i, rect in enumerate(kwargs['rect']):
                plt.gca().add_patch(
                    Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor=kwargs['rect_colors'][i],
                              facecolor='none'))
                # grid = draw_bounding_boxes(grid, kwargs['rect'], colors=kwargs['rect_colors'])

        plt.axis("off")

        if colormap is not None:
            plt.imshow(grid[0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap=colormap, vmin=grid.min(), vmax=grid.max())
        else:
            plt.imshow(grid.permute(1, 2, 0).numpy())

        # Use a custom title
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        plt.tight_layout()
        plt.margins(x=0, y=0)
        plt.show()

    else:
        return grid


def save_psnr_imgs(gt, cond, sample, save_dir, rotate = False, **kwargs):
    '''
    Input: Unnormalized, magnitude images (1, img_size, img_size)
    '''

    # Check if it is a np array
    gt = nu.check_type(gt, 'tensor')
    cond = nu.check_type(cond, 'tensor')
    sample = nu.check_type(sample, 'tensor')

    if rotate:
        gt = func.rotate(gt, 180)
        cond = func.rotate(cond, 180)
        sample = func.rotate(sample, 180)

    val_range = (gt.min(), gt.max())
    # Put the images in a grid and show them
    gtgrid = torchvision.utils.make_grid(gt,  normalize=True, value_range=val_range)
    condgrid = torchvision.utils.make_grid(cond, normalize=True, value_range=val_range)
    sample_imggrid = torchvision.utils.make_grid(sample, normalize=True, value_range=val_range)

    mean_psnr = metrics.psnr(gt.numpy(), sample.numpy())
    mean_psnr_cond = metrics.psnr(gt.numpy(), cond.numpy())

    save_grids(gtgrid, save_dir = os.path.join(save_dir, 'gt.png'))
    save_grids(condgrid, text = '{:.2f}'.format(mean_psnr_cond), save_dir=os.path.join(save_dir, 'cond.png'))
    save_grids(sample_imggrid, text = '{:.2f}'.format(mean_psnr), save_dir=os.path.join(save_dir, 'sample.png'))


def save_posteriors(x, save_dir, rotate = False, val_range=None, std=False, num_samples=1, **kwargs):
    '''
    Input: Unnormalized, magnitude images (batch_size, img_size, img_size)
    '''

    # Check if it is a tensor
    x = nu.check_type(x, 'tensor')

    # Make sure input only has 3 dimensions
    if len(x.shape) == 4:
        x = x.squeeze(1)

    if rotate:
        #x = func.rotate(x,180)
        x = func.vflip(x)

    if 'rect' in kwargs:
        if sum(kwargs['rect'][0]) < 1:
            rect = False
        else:
            rect = True
    else:
        rect = False


    #Create a separate folder for samples
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_samples):
        if val_range is None:
            val_range = (x.min(), x.max())

        if std:
            save_grids(x[i].unsqueeze(0), save_dir = os.path.join(save_dir, 'posterior{}.png').format(i),
                       std=std, value_range=val_range,
                       colorbar=True,
                       **kwargs)
        else:
            grid = torchvision.utils.make_grid(x[i].unsqueeze(0), normalize=True, value_range=val_range)
            save_grids(grid, save_dir = os.path.join(save_dir, 'posterior{}.png').format(i), std=std, **kwargs)

        if rect:
            for j, rect in enumerate(kwargs['rect']):
                if std:
                    save_grids(x[i, rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].unsqueeze(0), std=std,
                               save_dir=os.path.join(save_dir, 'posterior{}_zoom{}.png').format(i, j),
                               value_range=val_range,
                               colorbar=True)
                else:
                    grid_zoom = torchvision.utils.make_grid(x[i, rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]],  normalize=True, value_range=val_range)
                    save_grids(grid_zoom, std=std, save_dir = os.path.join(save_dir, 'posterior{}_zoom{}.png').format(i,j))



def save_grids(grid, std=False, value_range = None, colorbar = False, **kwargs):
    f = plt.figure()
    f.set_figheight(5)
    f.set_figwidth(5)

    # Turn the axis labels off
    plt.axis("off")

    if std:
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap=mpl.colormaps['viridis'], vmin=value_range[0], vmax=value_range[1])
        if colorbar:
            cbar = plt.colorbar(fraction=0.047, pad=0.01, format='%.0e', orientation='horizontal')
            cbar.ax.locator_params(nbins=3)

            # For vertical colorbar
            #cbar.ax.tick_params(labelsize='xx-large',rotation=90)
            # tl = cbar.ax.get_yticklabels()
            # tl[0].set_verticalalignment('bottom')
            # tl[-1].set_verticalalignment('top')

            # For horizontal colorbar
            cbar.ax.tick_params(labelsize='xx-large')
            tl = cbar.ax.get_xticklabels()
            tl[0].set_horizontalalignment('left')
            tl[-1].set_horizontalalignment('right')
    else:
        plt.imshow(grid.permute(1, 2, 0).numpy())

    if 'text' in kwargs:
        plt.text(20,30, kwargs['text'], backgroundcolor='w', fontsize='xx-large', fontweight='bold')

    #Add rectangles
    if 'rect' in kwargs:
        for i, rect in enumerate(kwargs['rect']):
            plt.gca().add_patch(
                Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2.5, edgecolor=kwargs['rect_colors'][i],
                          facecolor='none'))

    # Save the image
    if 'save_dir' in kwargs:
        plt.savefig(kwargs['save_dir'], bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.show()



def save_std_map(std_map, limits=None, colorbar=False, rotate = False, **kwargs):
    #Takes in the standard deviation map of unnnormalized, magnitude images (1, img_size, img_size)

    # Check if it is a tensor
    std_map = nu.check_type(std_map, 'tensor')

    # Put the images on the cpu
    std_map = std_map.detach().cpu()

    if rotate:
        std_map = func.rotate(std_map, 180)


    f = plt.figure()
    f.set_figheight(5)
    f.set_figwidth(5)

    if limits is not None:
        im = plt.imshow(std_map.permute(1, 2, 0).numpy(), cmap=mpl.colormaps['viridis'], vmin=limits[0], vmax=limits[1])
    else:
        im = plt.imshow(std_map.permute(1, 2, 0).numpy(), cmap=mpl.colormaps['viridis'])

    if colorbar:
        cbar = plt.colorbar(fraction=0.047, pad=0.01, format = '%.0e', orientation='horizontal')
        cbar.ax.locator_params(nbins=3)
        cbar.ax.tick_params(labelsize='xx-large')
        tl = cbar.ax.get_xticklabels()
        tl[0].set_horizontalalignment('left')
        tl[-1].set_horizontalalignment('right')

    plt.margins(x=0.01, y=0.01)

    plt.axis('off')

    if 'save_dir' in kwargs:
        # Save the image
        plt.savefig(os.path.join(kwargs['save_dir'],'std.png'), bbox_inches='tight', pad_inches=0)

    plt.tight_layout()
    plt.margins(x=0,y=0)
    plt.show()


def show_phase(plot_img, **kwargs):
    
    #Make sure the images are 4 dimensions
    if len(plot_img.shape) < 4:
        plot_img = plot_img.unsqueeze(0)
        
    #Get the phase
    phase = torch.arctan(plot_img[:,0,:,:] / plot_img[:,1,:,:])
    
    #Show the images
    show_img(phase.unsqueeze(1), val_range=(-torch.pi, torch.pi))
    

    
#Show absolute error map
def show_error_map(pred, gt, kspace = False, limits = None, title=None):

    err = torch.abs(pred - gt)

    f = plt.figure()
    f.set_figheight(10)
    f.set_figwidth(10)
    
    if kspace:
        err = torch.log(fastmri.complex_abs(pred-gt)+ 1e-9)
        err = err.unsqueeze(0)

    if limits is not None:
        plt.imshow(err.permute(1,2,0).numpy(), cmap =  mpl.colormaps['viridis'], vmin=limits[0], vmax=limits[1])
    else:
        plt.imshow(err.permute(1,2,0).numpy(), cmap =  mpl.colormaps['viridis'])
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()


def show_kspace(x, maps=None, val_range=(0,0.1)):
    # x is a multicoil image

    # Get the singlecoil image if not already
    if maps is not None:
        x = nu.multicoil2single(x, maps)

    #Show the kspace of the image
    k = fastmri.fft2c(nu.format_multicoil(x, chans=False))

    # Get the magnitude of the kspace
    k_mag = fastmri.complex_abs(k)

    # Get the phase of the kspace
    k_phase = torch.atan(k[:,:,:,:,0] / k[:,:,:,:,1])

    show_img(torch.log(k_mag + 1e-15))
    show_img(k_mag+1e-15, val_range=val_range)
    show_img(k_phase)

    


    

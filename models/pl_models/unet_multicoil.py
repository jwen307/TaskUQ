#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import torch
import numpy as np
import pytorch_lightning as pl

import sys


sys.path.append('../../')
from util import viz, network_utils, torch_losses
from models.networks.misc_nets import unet
import fastmri



#%% PyTorch Lightning  implementation
class UNetMulticoil(pl.LightningModule):
    
    def __init__(self, input_dims, build_func, config):
        '''
        flows: FrEIA Invertible Module
        build_bij_func: Function to build the bijective network
        net_args: Dictionary of arguments for the build_inj_func and build_bij_func and training args
        '''
        super().__init__()
        
        self.save_hyperparameters()
        
        self.input_dims = input_dims
        self.build_func = build_func
        self.config = config
        self.data_consistency = config['unet_args']['data_consistency']

        # Get the acs size
        self.acs_size = config['data_args']['acs_size']

        #Build the network
        self.build()

        
    # Function to build the network
    def build(self):
        #Build the bijective network
        self.model = self.build_func(self.input_dims, **self.config['unet_args'])

    
                    
    #Pass an image into the unet
    def forward(self, c, give_dc=False, mask= None, norm_val= None, give_features = False):
        ''' 
        c: conditional info (b, 2*coils, h, w)
        '''
        if give_features:
            feats = self.model(c, give_features)
            return feats.float()
        else:
            pred = self.model(c, give_features)
        
        if give_dc:
            pred, _ = network_utils.get_dc_image(pred, c, mask, norm_val=norm_val)

        return pred.float()

    
    def reconstruct(self, c, maps=None, mask = None, norm_val=None, multicoil=False, rss = False):
        ''' 
        cond: (b,16,h,w)
        output: (b, 2, h, w)
        '''
        
        #Get the prediction (b, 16, h, w)
        recons = self.forward(c)
        
        #Get the data consistency if needed
        if self.data_consistency:
            pred, pred_k = network_utils.get_dc_image(recons, c, mask, norm_val=norm_val)
        else:
            pred = recons

        # Put in coil format
        pred = network_utils.format_multicoil(pred, chans=False)
            
        norm_val = network_utils.check_type(norm_val, 'tensor')
        
        #Get the singlecoil prediction
        pred = pred.cpu() * norm_val.reshape(-1,1,1,1,1).cpu()
        
        if multicoil:
            return pred

        if rss:
            pred = fastmri.rss_complex(pred,dim=1)

        else:
            pred = network_utils.multicoil2single(pred, maps)

        return pred
    
    def configure_optimizers(self):
        # TODO: Changed from RMSProp
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = self.config['unet_arg']['unet_lr'],
                                     weight_decay=0.0
                                     )

        schedulers = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 40, gamma = 0.1
        )
        
        
        return [optimizer], [schedulers]
    

    def training_step(self, batch, batch_idx):
        c = batch[0]        # zero-filled
        x = batch[1]        # ground truth
        masks = batch[2]    # masks
        norm_val = batch[3]      # normalizing value
        
        #Get the prediction
        pred = self.model(c)

        #Apply the data consistent loss if needed
        if self.data_consistency:
            dcloss = torch_losses.DCLossMulticoil()
            loss= dcloss(pred, x, masks, norm_val)
        else:
            #Get the loss
            loss = torch.nn.functional.l1_loss(pred, x)
        
        # Log the training loss
        self.log('unet_train_loss', loss)

        return loss

    
    
    def validation_step(self, batch, batch_idx):
        
        c = batch[0]
        x = batch[1]
        masks = batch[2]
        norm_val = batch[3]

        #Get the maps
        maps = network_utils.get_maps(c, self.acs_size, normalizing_val=norm_val)

        board = self.logger.experiment
        
        with torch.no_grad():
            pred = self.model(c)
            
            #Find validation loss
            if self.data_consistency:
                dcloss = torch_losses.DCLossMulticoil()
                loss= dcloss(pred, x, masks, norm_val)
            else:
                #Get the loss
                loss = torch.nn.functional.l1_loss(pred, x)

            self.log('unet_val_loss', loss, rank_zero_only=True)
            
            #Show some examples
            if batch_idx == 1:
                #Show the original images and reconstructed images
                gt_grid = viz.show_img(x.float().detach().cpu(), maps, return_grid=True)
                board.add_image('GT Images', gt_grid, self.current_epoch)

                #Get the reconstructed images
                recons = self.reconstruct(c, maps, masks, norm_val)

                #Show the images
                grid = viz.show_img(recons.float().detach().cpu(), maps, return_grid=True)
                board.add_image('Val Image', grid, self.current_epoch)
            

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import fastmri
import torch
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F

import sys
sys.path.append('../../')
from util import network_utils
from models.net_builds.build_classifier import build_classifier
from train.configs.config_simclr_classifier import Config


class SIMCLR(pl.LightningModule):
    
    def __init__(self, config):

        super().__init__()
        
        self.save_hyperparameters()
        self.config = config

        # Figure out the size of the inputs
        img_size = config['data_args']['img_size']
        if config['data_args']['challenge'] == 'singlecoil':
            if config['data_args']['complex']:
                # TODO: Changed this to include real, imaginary, and magnitude
                self.input_dims = [3, img_size, img_size]
            else:
                self.input_dims = [3, img_size, img_size]
        elif config['data_args']['challenge'] == 'multicoil':
            self.input_dims = [2 * config['data_args']['num_vcoils'], img_size, img_size]
            
        self.network_type = config['net_args']['network_type']
        self.challenge = config['data_args']['challenge']
        self.complex = config['data_args']['complex']

        # Get the acs size
        if 'acs_size' in config['data_args']:
            self.acs_size = config['data_args']['acs_size']
        else:
            self.acs_size = 13

        # Use RSS to combine coils
        self.rss = config['net_args']['rss']

        self.epochs = config['train_args']['epochs']
        self.temp = config['net_args']['temp']

        # Build the network
        self.build()
        
    # Function to build the network
    def build(self):
        # Build the network with pretrained weights
        self.net = build_classifier(self.network_type, contrastive=True, input_chans=self.input_dims[0])

        self.transform_mean = self.net.transform_mean
        self.transform_std = self.net.transform_std



    def preprocess_data(self,x, cond, std):
        #Combine the coil images
        if self.rss:
            x = fastmri.rss_complex(network_utils.chans_to_coils(x), dim=1).unsqueeze(1)
        else:
            # Get the maps
            maps = network_utils.get_maps(cond, self.acs_size, std)

            # Get the singlecoil prediction
            x = network_utils.multicoil2single(x, maps)

            # Get the magnitude image
            x = fastmri.complex_abs(x).unsqueeze(1)

        x = self.reformat(x)

        return x



    def reformat(self,x):
        #Expects images to be (batch, 1, img_size, img_size)

        # Repeat the image for RGB channels
        x = x.repeat(1, 3, 1, 1)

        # Normalize to be between 0 and 1
        flattened_imgs = x.view(x.shape[0], -1)
        min_val, _ = torch.min(flattened_imgs, dim=1)
        max_val, _ = torch.max(flattened_imgs, dim=1)
        x = (x - min_val.view(-1, 1, 1, 1)) / (max_val.view(-1, 1, 1, 1) - min_val.view(-1, 1, 1, 1))

        # Define transforms based on pretrained network
        transforms = torchvision.transforms.Normalize(
            mean=self.transform_mean,
            std=self.transform_std
        )

        x = transforms(x)


        return x

    def normalize(self, x):
        # Normalize to be between 0 and 1
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)


        return x




    def forward(self, x, cond=None, std=None):

        # if self.challenge=='multicoil':
        #     #Preprocess the data
        #     x = self.preprocess_data(x, cond, std).to(self.device)
        # else:
        if not self.complex:
            x = self.reformat(x).to(self.device)


        # Non-adversarial network
        feats = self.net.get_features(x)

        #Classify using the features
        x = self.net.get_contrastive_proj(feats)

        # Returns the logits
        return x
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.net.parameters(),
                                    lr=self.config['train_args']['lr'],
                                    weight_decay=1e-2
                                    )

        schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['train_args']['epochs'], eta_min=self.config['train_args']['lr']/50
        )

        return [optimizer], [schedulers]


    def info_nce_loss(self, batch, mode='train'):
        if self.challenge == 'multicoil':
            imgs = batch[1]
        else:
            imgs = batch[0]
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temp
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + '_loss', nll)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + '_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode + '_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode + '_acc_mean_pos', 1 + sim_argsort.float().mean())

        return nll

                
    def training_step(self, batch, batch_idx):

        return self.info_nce_loss(batch, mode='train')


    
    def validation_step(self, batch, batch_idx):

        self.info_nce_loss(batch, mode='val')




from robustness.attacker import AttackerModel

if __name__ == '__main__':
    #Get the configurations
    config = Config()
    config = config.config
    config['train_args']['freeze_feats'] = 'all'

    # Set the input dimensions
    img_size = config['train_args']['img_size']
    input_dims = [3, img_size, img_size]

    # Initialize the network
    model = SIMCLR(input_dims,
                             config
                             )

    x = torch.ones(5,1,320,320)

    model = AttackerModel(model)

    y = model(x)
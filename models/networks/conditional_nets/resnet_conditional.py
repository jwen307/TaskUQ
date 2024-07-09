#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resnet_conditional.py
    - Modules for defining the conditioning network
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, 2 * out_ch, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * out_ch, 2 * out_ch, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * out_ch, out_ch, 1, padding=0, stride=1))

        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1)

        self.final_activation = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # conv = self.conv_block(self.final_activation(x))
        conv = self.conv_block(x)

        res = self.residual(x)

        y = self.batch_norm(conv + res)
        # y = conv+res
        y = self.final_activation(y)

        return y

# Class for the condtional network used in the single-coil CNF
class ResNetCondNet(nn.Module):
    """
    Attributes
    ----------
    img_size
    downsample_levels: Number of downsampling layers
    cond_conv_channels: Number of channels in each conditional convolutional layer
    use_fc_block: Whether to use a fully-connected block at the end of the network
    cond_fc_size: Number of channels in the fully-connected block

    Methods
    -------
    forward(c)
        Compute the forward pass.
        
    """
    
    def __init__(self, img_size, downsample_levels=5,cond_conv_channels=[4, 16, 32, 64, 64, 32], use_fc_block=True,cond_fc_size=128):

        super().__init__()

        
        self.img_size = img_size
        self.dsl = downsample_levels

        

        # FBP and resizing layers
        self.unet_out_shape = 16

        self.img_size = img_size
        self.dsl = downsample_levels
        self.fc_cond_dim = cond_fc_size
        self.shapes = [self.unet_out_shape] + cond_conv_channels
        self.use_fc_block = use_fc_block
        
        num_chs =  self.img_size[0]
        
        levels = []

        for i in range(self.dsl):
            levels.append(self.create_subnetwork(ds_level=i, dsl_idx = i))

        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl-1))

        # Network that preprocesses the input before passing to the CNNs
        self.preprocessing_net = nn.Sequential(
                ResBlock(in_ch=num_chs, out_ch=8),
                ResBlock(in_ch=8, out_ch=8), 
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1),
                ResBlock(in_ch=8, out_ch=16),
                ResBlock(in_ch=16, out_ch=16),
                nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),                            
                ResBlock(in_ch=16, out_ch=self.unet_out_shape),
  
        )
        

        self.resolution_levels = nn.ModuleList(levels)

    def forward(self, c):
        """
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.
        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).
        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.
        """
            
        
        outputs = []

        # Pass through the preprocessing network
        c_unet = self.preprocessing_net(c)

        # Pass through each CNN to get the conditional inputs for the flow
        for i, m in enumerate(self.resolution_levels):
            outputs.append(m(c_unet))

        return outputs


    def create_subnetwork(self, ds_level, dsl_idx):

        modules = []
        
        for i in range(ds_level+1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=2))
        
            modules.append(ResBlock(in_ch=self.shapes[i+1], out_ch=self.shapes[i+1]))


        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[dsl_idx+1],
                                kernel_size=1))
        return nn.Sequential(*modules)



    def create_subnetwork_fc(self, ds_level):

        modules = []
        
        for i in range(ds_level+1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=2))
            
            modules.append(ResBlock(in_ch=self.shapes[i+1], out_ch=self.shapes[i+1]))


        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[ds_level+1],
                                kernel_size=1))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.fc_cond_dim,
                                kernel_size=1))
        modules.append(nn.AvgPool2d(5,5))
        modules.append(nn.Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)
    

    

    

class ResNetCondNetProg(nn.Module):
    """
    Class for our modified conditional network. This network takes the output from the pretrained UNet.
    The features from the UNet are passed through a single CNN and the outputs of each layer are taken as conditional
    inputs for the flow.


    Attributes
    ----------
    img_size
    downsample_levels: Number of downsampling layers
    cond_conv_channels: Number of channels in each conditional convolutional layer
    use_fc_block: Whether to use a fully-connected block at the end of the network
    cond_fc_size: Number of channels in the fully-connected block

    Methods
    -------
    forward(c)
        Compute the forward pass.

        
    """
    
    def __init__(self, img_size, downsample_levels=3, cond_conv_channels=[64,64,128], use_fc_block=True,cond_fc_size=128, unet_out_shape = 128):

        super().__init__()

        
        self.img_size = img_size
        self.dsl = downsample_levels
        self.img_size = img_size
        self.unet_out_shape = unet_out_shape #Number of channels in the output of the UNet
        self.fc_cond_dim = cond_fc_size
        self.shapes = [self.unet_out_shape] + cond_conv_channels
        self.use_fc_block = use_fc_block
        
        
        # Define the layers of the conditional network CNN. Note: there is a single CNN
        levels = self.create_subnetwork(ds_level=self.dsl, dsl_idx=self.dsl)

        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl-1))

        self.resolution_levels = nn.ModuleList(levels)

    def forward(self, c):
        """
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.
        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).
        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.
        """
            
        
        outputs = []

        x = c

        # Get the intermediate outputs of each layer of the CNN
        for i, m in enumerate(self.resolution_levels):
            x = m(x)
            outputs.append(x)
            
        return outputs


    def create_subnetwork(self, ds_level, dsl_idx):
        levels=[]

        # Create the CNN
        for i in range(ds_level):
            modules = []
            
            
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=2))
        
            modules.append(ResBlock(in_ch=self.shapes[i+1], out_ch=self.shapes[i+1]))

            modules.append(nn.Conv2d(in_channels=self.shapes[i+1], 
                            out_channels=self.shapes[i+1], 
                            kernel_size=1))

            levels.append(nn.Sequential(*modules))

        return levels



    def create_subnetwork_fc(self, ds_level):

        modules = []
        
        for i in range(ds_level+1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i], 
                                    out_channels=self.shapes[i+1], 
                                    kernel_size=3, 
                                    padding=1, 
                                    stride=2))
            
            modules.append(ResBlock(in_ch=self.shapes[i+1], out_ch=self.shapes[i+1]))


        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.shapes[ds_level+1],
                                kernel_size=1))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level+1],
                                out_channels=self.fc_cond_dim,
                                kernel_size=1))
        modules.append(nn.AvgPool2d(5,5))
        modules.append(nn.Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)


class ResNetCondNetProgLatent(nn.Module):
    """
    Class for our modified conditional network. This network takes the output from the pretrained UNet.
    The features from the UNet are passed through a single CNN and the outputs of each layer are taken as conditional
    inputs for the flow.


    Attributes
    ----------
    img_size
    downsample_levels: Number of downsampling layers
    cond_conv_channels: Number of channels in each conditional convolutional layer
    use_fc_block: Whether to use a fully-connected block at the end of the network
    cond_fc_size: Number of channels in the fully-connected block

    Methods
    -------
    forward(c)
        Compute the forward pass.


    """

    # TODO: Main change, downsample levels include the downsampling that happens in the autoencoder, but only give the outputs
    # of the layers that are used in the flow

    def __init__(self, img_size, downsample_levels=3, latent_downsamples = 2, cond_conv_channels=[64, 64, 128], use_fc_block=True,
                 cond_fc_size=128, unet_out_shape=128):

        super().__init__()

        self.img_size = img_size
        self.dsl = downsample_levels + latent_downsamples
        self.unet_out_shape = unet_out_shape  # Number of channels in the output of the UNet
        self.fc_cond_dim = cond_fc_size
        self.shapes = [self.unet_out_shape] + [64 for _ in range(latent_downsamples)] + cond_conv_channels
        self.use_fc_block = use_fc_block

        if downsample_levels == 1:
            self.dsl -= 1

        # Number of downsamples that happen in the autoencoder
        self.latent_downsamples = latent_downsamples

        # Define the layers of the conditional network CNN. Note: there is a single CNN
        levels = self.create_subnetwork(ds_level=self.dsl, dsl_idx=self.dsl)

        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl - 1))

        self.resolution_levels = nn.ModuleList(levels)

    def forward(self, c):
        """
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.
        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).
        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.
        """

        outputs = []

        x = c

        # Get the intermediate outputs of each layer of the CNN
        for i, m in enumerate(self.resolution_levels):
            x = m(x)

            # Only output the results of the layers that are used in the flow
            if i >= self.latent_downsamples:
                outputs.append(x)

            # Case of a single scale in CNF
            if self.dsl==self.latent_downsamples:
                if i == self.latent_downsamples-1:
                    outputs.append(x)

        return outputs

    def create_subnetwork(self, ds_level, dsl_idx):
        levels = []

        # Create the CNN
        for i in range(ds_level):
            modules = []

            modules.append(nn.Conv2d(in_channels=self.shapes[i],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=3,
                                     padding=1,
                                     stride=2))

            modules.append(ResBlock(in_ch=self.shapes[i + 1], out_ch=self.shapes[i + 1]))

            modules.append(nn.Conv2d(in_channels=self.shapes[i + 1],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=1))

            levels.append(nn.Sequential(*modules))

        return levels

    def create_subnetwork_fc(self, ds_level):

        modules = []

        for i in range(ds_level + 1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=3,
                                     padding=1,
                                     stride=2))

            modules.append(ResBlock(in_ch=self.shapes[i + 1], out_ch=self.shapes[i + 1]))

        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level + 1],
                                 out_channels=self.shapes[ds_level + 1],
                                 kernel_size=1))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level + 1],
                                 out_channels=self.fc_cond_dim,
                                 kernel_size=1))
        modules.append(nn.AvgPool2d(5, 5))
        modules.append(nn.Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)


class ResNetCondNetProgLatent2(nn.Module):
    """
    Class for our modified conditional network. This network takes the output from the pretrained UNet.
    The features from the UNet are passed through a single CNN and the outputs of each layer are taken as conditional
    inputs for the flow.


    Attributes
    ----------
    img_size
    downsample_levels: Number of downsampling layers
    cond_conv_channels: Number of channels in each conditional convolutional layer
    use_fc_block: Whether to use a fully-connected block at the end of the network
    cond_fc_size: Number of channels in the fully-connected block

    Methods
    -------
    forward(c)
        Compute the forward pass.


    """

    # TODO: Main change, downsample levels include the downsampling that happens in the autoencoder, but only give the outputs
    # of the layers that are used in the flow

    def __init__(self, img_size, downsample_levels=3, latent_downsamples = 2, cond_conv_channels=[64, 64, 128], use_fc_block=True,
                 cond_fc_size=128, unet_out_shape=128):

        super().__init__()

        self.img_size = img_size
        self.dsl = downsample_levels + latent_downsamples
        self.unet_out_shape = unet_out_shape  # Number of channels in the output of the UNet
        self.fc_cond_dim = cond_fc_size
        self.shapes = [self.unet_out_shape] + [64 for _ in range(latent_downsamples)] + cond_conv_channels
        self.use_fc_block = use_fc_block

        #if downsample_levels == 1:
        #    self.dsl -= 1

        # Number of downsamples that happen in the autoencoder
        self.latent_downsamples = latent_downsamples

        # Define the layers of the conditional network CNN. Note: there is a single CNN
        levels = self.create_subnetwork(ds_level=self.dsl, dsl_idx=self.dsl)

        if self.use_fc_block:
            levels.append(self.create_subnetwork_fc(ds_level=self.dsl - 1))

        self.resolution_levels = nn.ModuleList(levels)

    def forward(self, c):
        """
        Computes the forward pass of the conditional network and returns the
        results of all building blocks in self.resolution_levels.
        Parameters
        ----------
        c : torch tensor
            Input to the conditional network (measurement).
        Returns
        -------
        List of torch tensors
            Results of each block of the conditional network.
        """

        outputs = []

        x = c

        # Get the intermediate outputs of each layer of the CNN
        for i, m in enumerate(self.resolution_levels):
            x = m(x)

            # Only output the results of the layers that are used in the flow
            if i >= self.latent_downsamples - 1:
                outputs.append(x)

            # Case of a single scale in CNF
            # if self.dsl==self.latent_downsamples:
            #     if i == self.latent_downsamples-1:
            #         outputs.append(x)

        return outputs

    def create_subnetwork(self, ds_level, dsl_idx):
        levels = []

        # Create the CNN
        for i in range(ds_level):
            modules = []

            modules.append(nn.Conv2d(in_channels=self.shapes[i],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=3,
                                     padding=1,
                                     stride=2))

            modules.append(ResBlock(in_ch=self.shapes[i + 1], out_ch=self.shapes[i + 1]))

            modules.append(nn.Conv2d(in_channels=self.shapes[i + 1],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=1))

            levels.append(nn.Sequential(*modules))

        return levels

    def create_subnetwork_fc(self, ds_level):

        modules = []

        for i in range(ds_level + 1):
            modules.append(nn.Conv2d(in_channels=self.shapes[i],
                                     out_channels=self.shapes[i + 1],
                                     kernel_size=3,
                                     padding=1,
                                     stride=2))

            modules.append(ResBlock(in_ch=self.shapes[i + 1], out_ch=self.shapes[i + 1]))

        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level + 1],
                                 out_channels=self.shapes[ds_level + 1],
                                 kernel_size=1))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Conv2d(in_channels=self.shapes[ds_level + 1],
                                 out_channels=self.fc_cond_dim,
                                 kernel_size=1))
        modules.append(nn.AvgPool2d(5, 5))
        modules.append(nn.Flatten())
        modules.append(nn.BatchNorm1d(self.fc_cond_dim))

        return nn.Sequential(*modules)
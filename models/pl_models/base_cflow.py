#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

base_cflow.py
    - PyTorch Lighting module for a base conditional flow
"""
import math
import torch
import numpy as np
import pytorch_lightning as pl

import sys
sys.path.append('../../')
from models.net_builds import build_multiscale_nets




class _BaseCFlow(pl.LightningModule):
    
    def __init__(self, config):

        '''
        configs: Configurations for the model
        '''
        super().__init__()
        
        self.save_hyperparameters()

        # Figure out the size of the inputs
        img_size = config['data_args']['img_size']
        if config['data_args']['challenge'] == 'singlecoil':
            if config['data_args']['complex']:
                self.input_dims = [2, img_size, img_size]
            else:
                self.input_dims = [1, img_size, img_size]
        elif config['data_args']['challenge'] == 'multicoil':
            self.input_dims = [2*config['data_args']['num_vcoils'], img_size, img_size]


        # Set the distribution
        self.distrib = config['flow_args']['distribution']

        self.latent_size = np.prod(self.input_dims)

        # Options for builds
        builds = [
            build_multiscale_nets.buildnetmcattn
        ]

        # Build the network
        self.build_bij_func = builds[config['flow_args']['build_num']]
        self.config = config

        self.use_transform_distrib = False
        self.num_layers = self.config['flow_args']['num_downsample'] - 1
        self.temp = 1.0


        
    # Function to build the network
    def build(self):
        #Build the bijective network
        self.flow, self.cond_net = self.build_bij_func(self.input_dims, **self.config['flow_args'])

        # Initialize the parameters
        self.init_params()

    def init_params(self):
        """
        Initialize the parameters of the model.

        Returns
        -------
        None.

        """
        # approx xavier
        #for p in self.cond_net.parameters():
        #    p.data = 0.02 * torch.randn_like(p) 
            
        for key, param in self.flow.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.02 * torch.randn(param.data.shape)
                # last convolution in the coeff func
                if len(split) > 3 and split[3][-1] == '4': 
                    param.data.fill_(0.)
                    
                    
    #rev = False is the normalizing direction, rev = True is generating direction
    def forward(self, x, cond_in, rev=False, **kwargs):
        
        #Get the conditional information
        c = self.cond_net(cond_in)
        
        #Normalizing direction
        if not rev:
            z, ldj = self.flow(x, c, rev=rev)
            
        #Generating direction
        else:
            z, ldj = self.flow(x, c, rev=rev)
            
        return z, ldj
    

    #Get latent vectors according to the specified distributions
    def sample_distrib(self, num_samples, temp=None):

        if self.use_transform_distrib:
            if self.qt is None:
                Exception('Need to fit the temperature first')
            self.temp = self.get_fitted_temps(num_samples)
            self.temp = self.temp.sort(dim=0, descending=True)[0]


        
        if self.distrib == 'radial':
            z = torch.randn(num_samples, self.latent_size, device=self.device)
            z_norm = torch.norm(z, dim=1)
            
            #Get the radius
            r = torch.abs(torch.randn((num_samples,1), device=self.device))
            
            #Normalize the vectors and then multiply by the radius
            z = z/z_norm.view(-1,1)*(r+temp-1.0)
            
        elif self.distrib == 'gaussian':
            z = torch.randn(num_samples, self.latent_size, device=self.device) * self.temp
            
        else:
            raise NotImplementedError()
            
        return z
        
    #Get the likelihood for a given set of latent vectors
    def get_nll(self,z, ldj, give_bpd=True, reduction='mean'):
        
        if self.distrib == 'gaussian':
            #Get the log probability of prior (assuming a Gaussian prior)
            log_pz = -0.5*torch.sum(z**2, 1) - (0.5 * np.prod(z.shape[1:]) * torch.log(torch.tensor(2*torch.pi)))
            
        elif self.distrib == 'radial':
            #Number of dimensions for each z
            n = torch.prod(torch.tensor(z.shape[1:]))
            
            #Get the log probability of prior (assuming a radial prior)
            log_pz = torch.log(torch.sqrt(2/(torch.pi * n))) - (n-1)*torch.log(torch.norm(z, dim=1)) - 0.5*torch.sum(z**2, 1)
            
        else:
            raise NotImplementedError()
            
        if self.training:
            self.log('log_pz', log_pz.mean(), sync_dist=True)
            self.log('ldj', ldj.mean(), sync_dist=True)

        #Get the log likelihood
        log_px = log_pz + ldj 
        nll = -log_px
        
        #Get the bits per dimension if needed
        if give_bpd:
            bpd = nll / (np.prod(z.shape[1:]) * np.log(2))
            #print('bpd: {0}'.format(bpd.mean()))
            return bpd.mean() if reduction == 'mean' else bpd
        
        return nll.mean() if reduction == 'mean' else nll


    def get_log_pz(self, z):
        if self.distrib == 'gaussian':
            # Get the log probability of prior (assuming a Gaussian prior)
            log_pz = -0.5 * torch.sum(z ** 2, 1) - (0.5 * np.prod(z.shape[1:]) * torch.log(torch.tensor(2 * torch.pi)))

        elif self.distrib == 'radial':
            # Number of dimensions for each z
            n = torch.prod(torch.tensor(z.shape[1:]))

            # Get the log probability of prior (assuming a radial prior)
            log_pz = torch.log(torch.sqrt(2 / (torch.pi * n))) - (n - 1) * torch.log(
                torch.norm(z, dim=1)) - 0.5 * torch.sum(z ** 2, 1)

        return log_pz





























    


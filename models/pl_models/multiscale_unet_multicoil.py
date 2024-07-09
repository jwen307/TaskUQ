#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import torch
import numpy as np
import pytorch_lightning as pl
import os
import time
import math
import fastmri
import sys

sys.path.append('../../')
from util import viz, network_utils, torch_losses


#Load modules for the UNet
from models.pl_models.base_cflow import _BaseCFlow
from models.pl_models.unet_multicoil import UNetMulticoil
from models.net_builds.build_unet import build0 as unetbuild
from models.networks.misc_nets import unet


class MulticoilCNF(_BaseCFlow):

    def __init__(self, config):
        '''
        configs: Configurations for the networks and training
        '''

        super().__init__(config)

        # Get the parameters
        self.null_proj = config['flow_args']['null_proj']
        self.acs_size = config['data_args']['acs_size']

        self.build()


        self.unet = unet.Unet(in_chans= self.input_dims[0],
                                out_chans= self.input_dims[0],
                                chans = config['unet_args']['chans'],
                                num_pool_layers=config['unet_args']['num_pool_layers'],
                                drop_prob=config['unet_args']['drop_prob']
                                )

        self.automatic_optimization = False



    # rev = False is the normalizing direction, rev = True is generating direction
    def forward(self, x, c, rev=False, posterior_sample=False, mask=None, norm_val=None):
        '''

        :param x: Either a latent vector or a ground truth image
        :param c: Zero-fileld image
        :param rev: False is normalizing direction, True is generating direction
        :param posterior_sample: Generate posterior samples (conditional info is the same)
        :param mask: Sampling mask
        :param maps: Sensitivity maps
        :param norm_val: Normalizing value
        :return: z: Latent vector or prediction, ldj: Log-determinant of the Jacobian
        '''
        num_samples = c.shape[0]

        # If we're looking for posterior samples, just pass in one of the conditionals (they're repeats)
        if posterior_sample:
            c = c[0].unsqueeze(0)
            norm_val = norm_val[0].unsqueeze(0)

        # Run the conditional data through the unet
        feats = self.unet(c, give_features=True)

        # Get the conditional information
        c = self.cond_net(feats)

        if posterior_sample:
            # Repeat the conditional information at each layer for posterior samples
            for k in range(len(c)):
                c[k] = c[k].repeat(num_samples, 1, 1, 1)

        # Normalizing direction
        if not rev:
            z, ldj = self.flow(x, c, rev=rev)

        # Generating direction
        else:
            z, ldj = self.flow(x, c, rev=rev)

        return z, ldj

    def reconstruct(self, c, num_samples, temp=1.0, check=False, maps=None, mask=None, norm_val=None,
                    split_num=None, rss=False, multicoil=False, dc = True):
        '''

        :param c: conditional images (zero-filled)
        :param num_samples: Number of posterior samples
        :param temp: Scale factor for the distribution
        :param check: Check for NaN reconstructions?
        :param maps: Sensitivity maps
        :param mask: Sampling mask
        :param norm_val: Normalizing value
        :param split_num: Number to split the posterior sample generation by to fit in memory
        :param rss: Return the root-sum-of-squares reconstruction
        :param multicoil: Return multicoil reconstructions
        :return: (list) recon: List of reconstructions
        '''

        # Get the batch_size
        b = c.shape[0]

        #Collect the reconstructions
        recons = []

        # Figure out how to split the samples so it fits in memory
        if split_num is not None and num_samples > split_num:
            num_splits = math.ceil(num_samples / split_num)
        else:
            num_splits = 1

        total_samples = num_samples * 1.0

        # Process each conditional image separately
        for i in range(b):
            with torch.no_grad():

                all_splits = []

                # Generate posteriors in batches so they fit in memory
                for k in range(num_splits):
                    if split_num is not None:
                        # Check if there's less than 16 samples left
                        if (total_samples - k * split_num) < split_num:
                            num_samples = int(total_samples - k * split_num)
                        else:
                            num_samples = split_num

                    # Draw samples from the distribution
                    z = self.sample_distrib(num_samples, temp=temp)

                    # Repeat the info for each sample
                    c_in = c[i].unsqueeze(0).repeat(num_samples, 1, 1, 1)
                    norm_val_rep = norm_val[i].unsqueeze(0).repeat(num_samples, 1, 1, 1)
                    mask_in = mask[i].unsqueeze(0).repeat(num_samples, 1, 1, 1)

                    # Get the reconstructions
                    recon, _, = self.forward(z, c_in, rev=True,
                                                posterior_sample=True,
                                                mask=mask_in, norm_val=norm_val_rep,
                                                )

                    # Check if the reconstructions are valid
                    if check:
                        # Replace the invalid reconstructions
                        recon = self.check_recon(recon, c[i], temp, mask=mask_in, norm_val=norm_val[i])

                    # Apply data consistency
                    if dc:
                        recon, _ = network_utils.get_dc_image(recon, c_in,
                                                              mask=mask_in,
                                                              norm_val=norm_val_rep)

                    # Unnormalize
                    recon = network_utils.unnormalize(recon ,norm_val[i])

                    # Combine the coils
                    if not multicoil:
                        recon = network_utils.format_multicoil(recon, chans=False)
                        if rss:
                            recon = fastmri.rss_complex(recon, dim=1).unsqueeze(1)
                        else:
                            rep_maps = network_utils.check_type(maps[i], 'tensor').unsqueeze(0).repeat(num_samples, 1, 1, 1)
                            recon = network_utils.multicoil2single(recon, rep_maps)
                            recon = network_utils.get_magnitude(recon)

                    # Collect all the splits
                    all_splits.append(recon)

                recons.append(torch.cat(all_splits, dim=0))

        # Output is a list where each element is a tensor with the reconstructions for a single conditional image
        return recons


    # Make sure the reconstructions don't have NaN
    def check_recon(self, recon, cond, temp, mask=None, norm_val=None):

        for i in range(recon.shape[0]):
            num_nan = 0

            # Check if the reconstruction is valid
            while recon[i].abs().max() > 10 or torch.isnan(recon[i].abs().max()):
                with torch.no_grad():
                    # Draw samples from a distribution
                    z = self.sample_distrib(1, temp=temp)
                    recon[i], _ = self.forward(z, cond.unsqueeze(0), rev=True,
                                               mask=mask,
                                               norm_val=norm_val.unsqueeze(0))

                    num_nan += 1

                    if num_nan > 10:
                        if i == 0:
                            print('Cant find inverse for img')
                            #viz.show_img(cond.unsqueeze(0))
                            #viz.show_img(recon[i])
                        break

        return recon

    def get_MAP(self, c, mask=None, norm_val=None, epochs=5000, lr=0.0000002, weights=[1.0, 0.0, 0.0]):
        '''
        Function to find the MAP estimate
        :param c:
        :param mask:
        :param maps:
        :param norm_val:
        :param epochs:
        :return:
        '''
        num_samples = c.shape[0]

        with torch.no_grad():

            # Start at a posterior image
            z = self.sample_distrib(1, temp=1.0).to(self.device)
            x, ldj = self(z, c, rev=True)

            # Get just the subsampled k-space and optimize on that
            # Go to kspace
            k = fastmri.fft2c(network_utils.unnormalize(network_utils.format_multicoil(x,chans=False), norm_val))
            inv_mask = mask[0, :, 0].cpu() * -1 + 1
            # Get the A matrix for subsampling
            A = torch.eye(x.shape[-1]).cuda()
            A = A[:, inv_mask > 0]

            # Get the subsampled kspace
            ksub = torch.matmul(network_utils.format_multicoil(k, chans=True), A)

        # Define the optimization
        ksub = ksub.requires_grad_(True)
        opt = torch.optim.Adam([ksub], lr=lr)

        xbest = x * 1.0
        loss_best = 100

        for i in range(epochs):
            # Zero the gradient
            opt.zero_grad()

            # Get the image
            k = torch.matmul(ksub, A.T)
            x = network_utils.normalize(fastmri.ifft2c(network_utils.format_multicoil(k,chans=False)), norm_val)
            x = network_utils.format_multicoil(x,chans=True)

            # Normalizing direction
            z, ldj = self(x, c, rev=False)

            # Get the loss
            bpd = self.get_nll(z, ldj, give_bpd=True)
            kurt = self.calc_kurtosis(z, loss_type='l2')
            shell = torch.abs(z.norm() - torch.sqrt(torch.tensor(len(z.flatten()))))
            #loss = self.get_nll(z, ldj, give_bpd=True)
            loss = weights[0] * bpd + weights[1]*kurt + weights[2]*shell
            #print('Epoch: {0}, Loss: {1}'.format(i, loss))
            print('Epoch: {0}, Loss: {1}, BPD: {2}, Kurt: {3}, Norm: {4}'.format(i, loss, bpd, kurt, z.norm()))

            if loss < loss_best:
                xbest = x.detach() * 1.0
                loss_best = loss.detach().cpu()

            # Backpropagate and update
            loss.backward()
            opt.step()

        return xbest.detach()

    def calc_kurtosis(self, z, loss_type='l2', reduce='sum'):
        vectors = z.flatten()

        # Calculate the kurtosis
        mean = torch.mean(vectors)
        var = torch.mean(torch.pow(vectors - mean,
                                   2.0))  # This is the biased estimate of the variance. torch.var is unbiased. Biased estimate lines up with scipy.kurtosis
        std = torch.pow(var, 0.5)

        kurtosis = torch.mean(torch.pow((vectors - mean) / std, 4.0)) - 3

        return kurtosis.abs() if loss_type == 'l1' else torch.pow(kurtosis, 2)



    def reconstruct_MAPS(self, cond, maps=None, mask=None, norm_val=None,
                    rss=False, multicoil=False, epochs=500, lr = 2e-7 ,weights = [1.0,0.0,0.0]):
        '''
        cond: (b,c,h,w)
        num_samples: Number of samples to draw from the latent distribution
        '''

        # Get the batch_size
        b = cond.shape[0]

        recons = []


        for i in range(b):

            # Get the MAP reconstruction
            recon = self.get_MAP(cond[i].unsqueeze(0), mask=mask, norm_val=norm_val[i], epochs=epochs, lr=lr, weights = weights)

            # Get the data consistent image
            recon, _ = network_utils.get_dc_image(recon, cond[i].unsqueeze(0),
                                                  mask=mask,
                                                  norm_val=norm_val[i].unsqueeze(0))

            # Unnormalize
            recon = network_utils.unnormalize(recon, norm_val[i])

            # Get the magnitude image
            if not multicoil:
                recon = network_utils.format_multicoil(recon, chans=False)

                # Get the magnitude predictions
                if rss:
                    recon = fastmri.rss_complex(recon, dim=1)
                else:
                    rep_maps = network_utils.check_type(maps[i], 'tensor').unsqueeze(0)
                    recon = network_utils.multicoil2single(recon, rep_maps)
                    recon = network_utils.get_magnitude(recon)

            recons.append(recon)

        return recons


    def configure_optimizers(self):

        cnf_opt = torch.optim.Adam(
            list(self.flow.parameters()) + list(self.cond_net.parameters()) + list(self.unet.parameters()),
            lr=self.config['train_args']['lr'],
            weight_decay=1e-7
            )

        pretrain_opt = torch.optim.Adam(
            self.unet.parameters(),
            lr=self.config['unet_args']['unet_lr'],
            weight_decay=1e-7
        )

        return pretrain_opt, cnf_opt


    def training_step(self, batch, batch_idx):

        #Get the optimizers
        pretrain_opt, cnf_opt = self.optimizers()

        # Get all the inputs
        c = batch[0].to(self.device)
        x = batch[1].to(self.device)
        masks = batch[2].to(self.device)
        norm_val = batch[3].to(self.device)

        # Pretraining
        if self.config['train_args']['pretrain_unet'] and (self.current_epoch < self.config['unet_args']['unet_num_epochs']):
            pred = self.unet(c)

            # Check if using data consistency with UNet
            if self.config['unet_args']['data_consistency']:
                dcloss = torch_losses.DCLossMulticoil()
                loss = dcloss(pred, x, masks, norm_val)
            else:
                # Get the loss
                loss = torch.nn.functional.l1_loss(pred, x)

            # Log the training loss
            self.log('unet_train_loss', loss)

            # Backpropagate and update
            pretrain_opt.zero_grad()
            self.manual_backward(loss)
            pretrain_opt.step()

        # Full CNF training
        else:
            # Apply inverse mask to GT
            if self.null_proj:
                x = network_utils.apply_mask(x, masks, norm_val=norm_val, inv_mask=True)

            # Pass through the CNF
            z, ldj = self(x, c, rev=False)

            # Find the negative log likelihood
            loss = self.get_nll(z, ldj, give_bpd=True)

            # Log the training loss
            self.log('train_loss', loss, prog_bar=True)

            # Backpropagate and update
            cnf_opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(cnf_opt, gradient_clip_val=1.0)
            cnf_opt.step()

        return loss

    def validation_step(self, batch, batch_idx):

        c = batch[0]
        x = batch[1]
        masks = batch[2]
        norm_val = batch[3]

        board = self.logger.experiment

        with torch.no_grad():

            # Pretraining
            if self.config['train_args']['pretrain_unet'] and (self.current_epoch < self.config['unet_args']['unet_num_epochs']):
                pred = self.unet(c)

                # Find validation loss
                if self.config['unet_args']['data_consistency']:
                    dcloss = torch_losses.DCLossMulticoil()
                    loss = dcloss(pred, x, masks, norm_val)
                else:
                    # Get the loss
                    loss = torch.nn.functional.l1_loss(pred, x)

                self.log('unet_val_loss', loss, sync_dist=True)

                # Just log some number so it doesn't conflict with the model checkpointing
                self.log('val_bpd', 10, sync_dist=True)


            # Full CNF training
            else:
                if self.null_proj:
                    xs = network_utils.apply_mask(x, masks, norm_val=norm_val, inv_mask=True)

                else:
                    xs = x

                z, ldj = self(xs, c, rev=False)

                # Find the negative log likelihood
                loss = self.get_nll(z, ldj, give_bpd=True)
                self.log('val_bpd', loss, sync_dist=True)

                # Show some example images
                if batch_idx == 1:
                    # TODO: Note: the sensitivity map estimation messes with multi-gpu checkpointign so do RSS here
                    # Show the original images and reconstructed images
                    gt_grid = viz.show_img(x.float().detach().cpu(), return_grid=True, rss=True)
                    board.add_image('GT Images', gt_grid, self.current_epoch)

                    recons = self.reconstruct(c, num_samples=2, mask=masks, norm_val=norm_val, rss=True)

                    # Concatenate the list
                    recons = torch.cat(recons, dim=0)

                    # Show the images
                    grid = viz.show_img(recons.float().detach().cpu(), return_grid=True)
                    board.add_image('Full Val Image', grid, self.current_epoch)



# Uncomment to test model

#%% Test out the model to make sure it runs
if __name__ == '__main__':
    from train.configs.config_cinn_unet_multicoil import Config
    from datasets.fastmri_multicoil import FastMRIDataModule
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Get the configurations
    conf = Config()
    configs = conf.config

    if configs['data_args']['mri_type'] == 'brain':
        base_dir = "/storage/fastMRI_brain/data/"
    elif configs['data_args']['mri_type'] == 'knee':
        base_dir = "/storage/fastMRI/data/"
    else:
        raise Exception("Please specify an mri_type in configs")

    # Get the data
    data = FastMRIDataModule(base_dir,
                             num_data_loader_workers=4,
                             **configs['data_args'],
                             )
    data.prepare_data()
    data.setup()


    # Initialize the network
    model = MulticoilCNF(configs)

    dataset = data.val
    model = model.cuda()
    model = model.eval()
    samp_num=2004

    # Get the data
    cond = dataset[samp_num][0].unsqueeze(0).to(model.device)
    gt = dataset[samp_num][1].unsqueeze(0).to(model.device)
    mask = dataset[samp_num][2].to(model.device)
    norm_val = torch.tensor(dataset[samp_num][3]).unsqueeze(0).to(model.device)
    maps = network_utils.get_maps(cond, model.acs_size, norm_val)

    with torch.no_grad():
        z, ldj = model(gt, cond, rev=False, mask=mask, norm_val=norm_val[0])
        recon, ldj1 = model(z, cond, rev=True, mask=mask, norm_val=norm_val[0])
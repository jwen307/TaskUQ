

from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import sys

import fastmri
from fastmri.data import transforms
#from fastmri.models import VarNet

sys.path.append('../../')
from util import viz, network_utils, torch_losses
from models.networks.misc_nets.varnet import VarNet
from datasets.masks.mask import get_mask, apply_mask

class VarNetModule(pl.LightningModule):
    """
    VarNet training module.

    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmri.SSIMLoss()

    def forward(self, masked_kspace, mask, num_low_frequencies):
        return self.varnet(masked_kspace, mask, num_low_frequencies)

    def reconstruct(self, c, mask, norm_val, **kwargs):
        # Get the masked k-space from the zero_filled image
        masked_kspace = fastmri.fft2c(
            network_utils.format_multicoil(network_utils.unnormalize(c, norm_val), chans=False))
        # Re-zero out
        masked_kspace = apply_mask(masked_kspace, mask)

        # output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
        output = self(masked_kspace, mask.to(torch.bool), num_low_frequencies=None)

        return output


    def training_step(self, batch, batch_idx):
        c = batch[0]  # zero-filled
        x = batch[1]  # ground truth
        masks = batch[2].unsqueeze(1)  # masks
        norm_val = batch[3]  # normalizing value

        # Get the masked k-space from the zero_filled image
        masked_kspace = fastmri.fft2c(network_utils.format_multicoil(network_utils.unnormalize(c, norm_val), chans=False))
        # Re-zero out
        masked_kspace = apply_mask(masked_kspace, masks)

        # Get the magnitude image
        target = fastmri.rss_complex(network_utils.format_multicoil(network_utils.unnormalize(x, norm_val), chans=False), dim=1)

        #output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
        output = self(masked_kspace, masks.to(torch.bool), num_low_frequencies=None)

        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=target.amax(dim=(-2, -1))
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        c = batch[0]  # zero-filled
        x = batch[1]  # ground truth
        masks = batch[2].unsqueeze(1)  # masks
        norm_val = batch[3]  # normalizing value

        board = self.logger.experiment

        # Get the masked k-space from the zero_filled image
        masked_kspace = fastmri.fft2c(
            network_utils.format_multicoil(network_utils.unnormalize(c, norm_val), chans=False))
        # Re-zero out
        masked_kspace = apply_mask(masked_kspace, masks)

        # Get the magnitude image
        target = fastmri.rss_complex(network_utils.format_multicoil(network_utils.unnormalize(x, norm_val), chans=False), dim=1)

        # output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
        output = self(masked_kspace, masks.to(torch.bool), num_low_frequencies=None)

        target, output = transforms.center_crop_to_smallest(target, output)
        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=target.amax(dim=(-2, -1))
        )

        self.log("val_loss", loss)

        # Show some examples
        if batch_idx == 1:
            # Show the original images and reconstructed images
            gt_grid = viz.show_img(target.unsqueeze(1).float().detach().cpu(), return_grid=True)
            board.add_image('GT Images', gt_grid, self.current_epoch)

            # Show the images
            grid = viz.show_img(output.unsqueeze(1).float().detach().cpu(), return_grid=True)
            board.add_image('Val Image', grid, self.current_epoch)



    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]



#%%  Test out model

if __name__ == '__main__':
    from train.configs.config_varnet import Config
    from datasets.fastmri_multicoil import FastMRIDataModule
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
                             batch_size=configs['train_args']['batch_size'],
                             num_data_loader_workers=4,
                             **configs['data_args'],
                             )
    data.prepare_data()
    data.setup()

    # Initialize the network
    model = VarNetModule(**configs['model_args'])

    dataset = data.val
    model = model.cuda()
    model = model.eval()
    samp_num = 20

    # Get the data
    cond = dataset[samp_num][0].unsqueeze(0).to(model.device)
    gt = dataset[samp_num][1].unsqueeze(0).to(model.device)
    mask = dataset[samp_num][2].to(model.device).unsqueeze(0).unsqueeze(0)
    norm_val = torch.tensor(dataset[samp_num][3]).unsqueeze(0).to(model.device)
    #maps = network_utils.get_maps(cond, model.acs_size, norm_val)

    with torch.no_grad():
        # Get the masked k-space from the zero_filled image
        masked_kspace = fastmri.fft2c(network_utils.format_multicoil(network_utils.unnormalize(cond, norm_val), chans=False))
        # Re-zero out
        masked_kspace = apply_mask(masked_kspace, mask)

        output = model(masked_kspace, mask.to(torch.bool), num_low_frequencies=None)


    # Get a batch
    batch = next(iter(data.val_dataloader()))

    mask = batch[2].unsqueeze(1).to(model.device)

    with torch.no_grad():
        # Get the masked k-space from the zero_filled image
        masked_kspace = fastmri.fft2c(network_utils.format_multicoil(network_utils.unnormalize(batch[0], batch[3]), chans=False)).to(model.device)
        # Re-zero out
        masked_kspace = apply_mask(masked_kspace, mask)

        output = model(masked_kspace, mask.to(torch.bool), num_low_frequencies=None)




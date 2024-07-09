#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import os

import fastmri
import torch
import pytorch_lightning as pl
import torchvision
import torchmetrics
from sklearn.metrics import balanced_accuracy_score

from robustness.model_utils import DummyModel
from robustness.attacker import AttackerModel

import sys
sys.path.append('../../')
from util import network_utils, helper
from models.net_builds.build_classifier import build_classifier
from train.configs.config_binary_classifier import Config



class BinaryClassifier(pl.LightningModule):
    
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

        # Freeze a set of the features before supervised training
        self.freeze_feats = config['net_args']['freeze_feats']

        # Weight decay
        if 'weight_decay' in config['train_args']:
            self.weight_decay = config['train_args']['weight_decay']
        else:
            self.weight_decay = 0

        # Use adversarial training
        self.adversarial = config['net_args']['adversarial']

        # Define the loss function
        self.bce_weight = config['net_args']['bce_weight']
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.bce_weight], device=self.device))

        # Define adversarial loss parameters
        self.adv_kwargs = config['adversarial_args']

        # Define different metrics to measure
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.train_auroc = torchmetrics.AUROC(task='binary')

        # Check if you want to use a pretrained network (i.e. contrastive learning)
        self.pretrained_ckpt = config['net_args']['pretrained_ckpt'] if 'pretrained_ckpt' in config['net_args'] else None

        # Build the network
        self.build()

        # Wrap with adversarial model if needed
        if self.adversarial:
            self.net = AttackerModel(DummyModel(self.net), self.net.t)

        if 'use_posteriors' in config['train_args']:
            self.use_posteriors = config['train_args']['use_posteriors']
            if self.use_posteriors:
                config_file = os.path.join(config['train_args']['cnf_dir'], 'configs.pkl')
                cnf_config = helper.read_pickle(config_file)
                model_type = cnf_config['flow_args']['model_type']
                ckpt_name = 'last.ckpt' if config['train_args']['load_last_ckpt'] else 'best.ckpt'
                ckpt = os.path.join(config['train_args']['cnf_dir'],
                                    'checkpoints',
                                    ckpt_name)
                self.cnf = helper.load_model(model_type, cnf_config, ckpt)
                self.cnf.eval()
                for param in self.cnf.parameters():
                    param.requires_grad = False

        else:
            self.use_posteriors = False
            self.cnf = None


    # Function to build the network
    def build(self):
        # Build the network with pretrained weights
        if self.pretrained_ckpt is not None:
            print('Loading pretrained network')
            ckpt = os.path.join(self.pretrained_ckpt,
                                'checkpoints',
                                'best_val_loss.ckpt')

            # Get the configuration file
            config_file = os.path.join(self.pretrained_ckpt, 'configs.pkl')
            config_pretrained = helper.read_pickle(config_file)

            # Load the model
            model = helper.load_model(config_pretrained['net_args']['model_type'], config_pretrained, ckpt)

            if config_pretrained['net_args']['model_type'] == 'MAE':
                self.net = model.get_classifier()

            else:
                self.net = model.net

        else:
            self.net = build_classifier(self.network_type, input_chans=self.input_dims[0])

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

        # Apply the transforms (transforms are applied in adversarial framework so don't apply here)
        if not self.adversarial:
            x = transforms(x)

        return x

    def normalize(self, x):
        # Normalize to be between 0 and 1
        flattened_imgs = x.view(x.shape[0], -1)
        min_val, _ = torch.min(flattened_imgs, dim=1)
        max_val, _ = torch.max(flattened_imgs, dim=1)
        x = (x - min_val.view(-1, 1, 1, 1)) / (max_val.view(-1, 1, 1, 1) - min_val.view(-1, 1, 1, 1))


        return x



    def forward(self, x, cond=None, std=None, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):


        if not self.complex:
            x = self.reformat(x).to(self.device)


        if self.adversarial:
            if target is not None:
                x, adv = self.net(x, target, make_adv, with_latent, fake_relu, no_relu, with_image=True,
                                  **attacker_kwargs)

                return x.flatten(), adv

            else:
                x = self.net(x, with_image=False).flatten()


        # Non-adversarial network
        else:
            #Get the extracted features
            if self.freeze_feats is not None: #and self.training:
                self.net.feature_extractor.eval()
                with torch.no_grad():
                    feats = self.net.get_features(x)

            else:
                feats = self.net.get_features(x)

            #Classify using the features
            x = self.net.classify(feats)

        # Returns the logits
        return x
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),
                                        lr=self.config['train_args']['lr'],
                                        weight_decay=self.weight_decay
                                        )

        schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['train_args']['epochs'], eta_min=self.config['train_args']['lr'] / 50
        )

        return [optimizer], [schedulers]

                
    def training_step(self, batch, batch_idx):

        # Get the inputs and targets
        x = batch[0]
        y = batch[-1]

        if self.adversarial:
            # Do the adversarial prediction
            # Note: since targeted is False by default in adv_kwargs, this does gradient ascent to maximize the loss
            ypred, im_adv = self(x, target=y.unsqueeze(1), make_adv=True, **self.adv_kwargs)

        else:
            #Get the prediction
            ypred = self(x)


        ypred = ypred.flatten()

        loss = self.loss_fn(ypred,y.float())

        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.train_auroc(ypred.detach(), y)
        self.log('Train AUROC', self.train_auroc, on_step=True)

        return loss


    
    def validation_step(self, batch, batch_idx):

        # Get the inputs and targets
        x = batch[0]
        y = batch[-1]

        # Get the prediction
        ypred = self(x)

        ypred = ypred.flatten()


        loss = self.loss_fn(ypred, y.float())

        self.log('val_loss', loss, on_epoch=True)

        self.val_accuracy(ypred.detach(),y)
        self.log('Val Accuracy', self.val_accuracy, on_epoch=True)
        self.val_precision(ypred.detach(),y)
        self.log('Val Precision', self.val_precision, on_epoch=True)
        self.val_recall(ypred.detach(),y)
        self.log('Val Recall', self.val_recall, on_epoch=True)
        self.val_auroc(ypred.detach(),y)
        self.log('Val AUROC', self.val_auroc, on_epoch=True)

        # Find the balanced accuracy
        bal_acc = balanced_accuracy_score(y.cpu().int(), torch.round(torch.sigmoid(ypred.detach())).cpu().int())
        self.log('Val Bal Accuacy', bal_acc, on_epoch=True)



if __name__ == '__main__':
    #Get the configurations
    config = Config()
    config = config.config
    config['train_args']['freeze_feats'] = 'all'

    # Set the input dimensions
    img_size = config['train_args']['img_size']
    input_dims = [3, img_size, img_size]

    # Initialize the network
    model = BinaryClassifier(input_dims,
                             config
                             )

    x = torch.ones(5,1,320,320)

    model = AttackerModel(model)

    y = model(x)
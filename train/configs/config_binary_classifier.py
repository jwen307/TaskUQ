#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import torch
import torchvision.transforms as transforms

class Config():
    
    def __init__(self):
        
        self.config = {
            'data_args':
                {
                    'mri_type': 'knee',
                    'img_size': 320,
                    'challenge': "singlecoil",
                    'complex': False,
                    'scan_type': 'CORPD_FBK',  # 'CORPD_FBK' or 'CORPDFS_FBK'
                    'slice_range': None,  # [0, 8], None, 0.8
                    'specific_label': ['Meniscus Tear'],
                    'augmented': True,
                    'num_vcoils': 8,
                    'acs_size': 9,
                    'accel_rate': 16,
                    'mask_type': 'knee',
                    'mask_box_augment': False,
                },
                
			'train_args':
                {
                    'lr': 1e-4,
                    'batch_size': 128,
                    'epochs': 100,
                    'weight_decay': 1e-2,
                    'use_posteriors': False,
                    'load_last_ckpt': False,
                },

            'net_args':
                {
                    'rss': True,
                    'freeze_feats' : 'None', #None, 'some', 'all'
                    'network_type': 'resnet50', #'resnet18', 'vgg11', 'swin'
                    'adversarial': True,
                    'bce_weight': 1,
                    'pretrained_ckpt': '/storage/SIMCLR/version_0', # Directory of the SIMCLR model
                },

            'adversarial_args':
            {
                'constraint': '2',  # use L2-PGD
                'eps': 1.5,  # L2 radius around original image
                'step_size': 0.05,  # step size for PGD
                'iterations': 10,
                'do_tqdm': False,
                'custom_loss': custom_adv_loss,
                'use_best': False
            },

        }



def custom_adv_loss(model, inp, target):
    #Need to return a loss for each input so reduction = 'none'
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=target.device), reduction='none')
    pred = model(inp)
    loss = criterion(pred, target.float())
    #loss = self.loss_fn(pred, target.float())

    return loss, None



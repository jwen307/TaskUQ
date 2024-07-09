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
                    'augmented': False,
                    'contrastive': True, # Set augmented to be false if true
                    'num_vcoils': 8,
                    'acs_size': 9,
                    'accel_rate': 16,
                    'mask_type': 'knee'
                },
                
			'train_args':
                {
                    'lr': 2e-4, # 2e-4 for resnet50, 5e-5 for swin
                    'batch_size': 128,
                    'epochs': 500,
                },

            'net_args':
                {
                    'model_type': 'SIMCLR',
                    'rss': True,
                    'freeze_feats' : None, #None, 'some', 'all'
                    'network_type': 'resnet50', #'resnet18', 'vgg11', 'swin'
                    'temp': 0.07,
                },

        }



def custom_adv_loss(model, inp, target):
    #Need to return a loss for each input so reduction = 'none'
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1], device=target.device), reduction='none')
    pred = model(inp)
    loss = criterion(pred, target.float())
    #loss = self.loss_fn(pred, target.float())

    return loss, None



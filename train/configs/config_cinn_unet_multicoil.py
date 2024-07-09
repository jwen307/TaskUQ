#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Configuration file for training a conditional flow with a U-Net
"""

class Config():
    def __init__(self):

        self.config = {
            'data_args':
                {
                'mri_type': 'knee',  # brain or knee
                'center_frac': 0.08,
                'accel_rate': [2,4,8,16],
                'img_size': 320,
                'challenge': "multicoil",
                'complex': True, # if singlecoil, specify magnitude or complex
                'scan_type': ['CORPD_FBK'],  # Knee: 'CORPD_FBK' Brain: 'AXT2'
                'mask_type': 'prog',  # Options :'s4', 'default', 'center_aug'
                'num_vcoils': 8,
                'acs_size': 9,  # 13 for knee, 32 for brain
                'slice_range': None,  # [0, 8], None
                },

            'train_args':
                {
                'lr': 1e-4,
                'batch_size': 4,
                'pretrain_unet': False
                },

            'flow_args':
                {
                'model_type': 'MulticoilCNF',
                'distribution': 'gaussian',
                'build_num': 0,

                # Flow parameters
                'num_downsample': 2,
                'cond_conv_chs': [64, 64],
                'downsample': 'squeeze',
                'num_blocks': 10,
                'use_fc_block': False,
                'num_fc_blocks': 2,
                'cond_fc_size': 64,
                'null_proj': True, #Use nullspace projection
                'subnet': 'conv3x3', #Select a subnetwork type (only works when build_num=2)
                'attn_type': 'imap', #Select an attention type (only works when build_num=4)
                'unet_out_shape': 256,
                },

            'unet_args':
                {
                'chans': 256,
                'num_pool_layers': 4,
                'drop_prob': 0.0,
                'data_consistency': True,
                'unet_num_epochs': 50,
                'unet_lr': 3e-3,
                }
        }




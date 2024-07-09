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
                'batch_size': 8,
                'model_type': 'VarNet',
                },

            'model_args':
                {
                'num_cascades': 12,
                'pools': 4,
                'chans': 18,
                'sens_pools': 4,
                'sens_chans': 8,

                'lr': 1e-4,
                'lr_step_size': 40,
                'lr_gamma': 0.1,
                'weight_decay': 0.0,
                },


        }




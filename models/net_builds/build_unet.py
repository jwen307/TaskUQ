#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

build_unet.py
    - Functions to build the unet architecture
"""


import FrEIA.framework as Ff
import FrEIA.modules as Fm

import sys
sys.path.append('../..')


from models.networks.misc_nets import unet


def build0(img_size=[1,320,320], **kwargs):
    
    net = unet.Unet(in_chans= img_size[0], 
                    out_chans= img_size[0], 
                    chans = kwargs['chans'], 
                    num_pool_layers=kwargs['num_pool_layers'],
                    drop_prob=kwargs['drop_prob']
                    )
    
    return net
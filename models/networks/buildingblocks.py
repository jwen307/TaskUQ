#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buildingblocks.py
    - Functions that define the building blocks of the normalizing flow
"""

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from .inn_modules import coupling, misc, op
from .subnets import subnetworks


def _add_downsample(nodes, downsample):
    """
    Downsampling operations.
    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsample : str
        Downsampling method. Currently there are three options: 'haar', 
        'reshape' and 'squeeze'.
    Returns
    -------
    None.
    """
    
    if downsample == 'haar':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.HaarDownsampling, 
                               {'rebalance':0.5, 'order_by_wavelet':True},
                               name='haar'))
    if downsample == 'reshape':
        nodes.append(Ff.Node(nodes[-1].out0, Fm.IRevNetDownsampling, {},
                               name='reshape'))
        
    if downsample == 'squeeze':
        nodes.append(Ff.Node(nodes[-1].out0, op.SqueezeFlow, {}, name='squeeze'))
        




def _add_fc_section(nodes, cond, num_blocks, coupling_type):
    """
    Add conditioned notes to the network.
    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    num_blocks: int
        Number of coupling blocks    
    cond : TYPE
        FrEIA condition note
    coupling: str
        Type of coupling used 
    Returns
    -------
    None.
    """
    
    for k in range(num_blocks):        
        if coupling_type == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnetworks.subnet_fc, 'clamp':1.5},
                             conditions = cond,
                             name="GLOWBlock_fc.{}".format(k)))
        else: 
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                             {'subnet_constructor':subnetworks.subnet_fc},
                             conditions = cond,
                             name="NICEBlock_fc.{}".format(k)))
        
        nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom, 
                                 {'seed':k}, 
                                 name='PermuteRandom_fc.{}'.format(k)))

def add_level_transition(nodes, downsampling_level):
    ''' 
    Adds a level transistion step. Used after a squeeze operation
    '''
    
    #Activation Normalization
    nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {}, name="ActNorm_{}".format(downsampling_level)))
    
    #1x1 Convolution
    nodes.append(Ff.Node(nodes[-1].out0, misc.Fixed1x1ConvOrthogonal, 
                        {}, 
                        name='1x1 Conv Transition {0}'.format(downsampling_level)))





def _add_conditioned_section_attn(nodes, downsampling_level, num_blocks, cond, coupling_type, act_norm, permutation_type,
                                 subnet_in=None, down_layer=0, attn_type='imap'):
    """
    New conditional block
    Add conditioned notes to the network.
    Parameters
    ----------
    nodes : TYPE
        Current nodes of the network.
    downsampling_level: int
        Current downsampling level
    num_blocks: int
        Number of coupling blocks
    cond : TYPE
        FrEIA condition note
    coupling_type: str
        Type of coupling used
    act_norm: bool
        whether to use act norm
    permutation_type: str
        which permutation to use
    Returns
    -------
    None.
    """

    for k in range(num_blocks):

        if act_norm:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.ActNorm, {},
                                 name="ActNorm_{}.{}.{}".format(downsampling_level, k, down_layer)))

        if permutation_type == "1x1":
            nodes.append(Ff.Node(nodes[-1].out0, misc.Fixed1x1ConvOrthogonal,
                                 {},
                                 name='1x1Conv_{}.{}.{}'.format(downsampling_level, k, down_layer)))
        elif permutation_type == "random":
            nodes.append(Ff.Node(nodes[-1].out0, Fm.PermuteRandom,
                                 {'seed': (k + 1) * (downsampling_level + 1)},
                                 name='PermuteRandom_{}.{}.{}'.format(downsampling_level, k, down_layer)))

        if k % 2 == 0:
            subnet = subnetworks.subnet_conv1x1
        else:
            subnet = subnetworks.subnet_conv3x3 if subnet_in is None else subnet_in

        if coupling_type == 'affine':
            nodes.append(Ff.Node(nodes[-1].out0, Fm.AffineCouplingOneSided,
                                 {'subnet_constructor': subnet, 'clamp': 1.5},
                                 conditions=cond,
                                 name="GLOWBlock_{}.{}.{}".format(downsampling_level, k, down_layer)))
        else:
            nodes.append(Ff.Node(nodes[-1].out0, Fm.NICECouplingBlock,
                                 {'subnet_constructor': subnet},
                                 conditions=cond,
                                 name="NICEBlock_{}.{}.{}".format(downsampling_level, k, down_layer)))


        if attn_type == 'imap':
            nodes.append(Ff.Node(nodes[-1].out0, coupling.iMap,
                                 {},
                                 name="iMap_{}.{}.{}".format(downsampling_level, k, down_layer)))

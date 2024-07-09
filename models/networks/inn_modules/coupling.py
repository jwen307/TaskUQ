#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coupling.py
    - Modules to define components of the normalizing flow
"""
import math
import torch
from typing import Callable
import torch.nn as nn

import FrEIA.modules as Fm
from typing import Callable, Union
from einops import rearrange
import numpy as np


   
#This layer takes the conditional information and uses it to find the bias and scale
class AffineInjector(Fm.InvertibleModule):
    
    def __init__(self, dims_in, dims_c=[], subnet_constructor: Callable = None):
        
        super().__init__(dims_in, dims_c)
        
        self.channels, self.h, self.w = dims_in[0]
        
        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])
        
        
        #Twice as many outputs because separate outputs for s and t
        self.subnet = subnet_constructor(self.condition_length,  self.channels*2)

        
    def forward(self, x, c=[], rev=False, jac = True):
        
        #x is passed as a list, so use x[0]
        x=x[0]
        

        #Pass the masked version in
        log_scale_shift = self.subnet(c[0])
        t = log_scale_shift[:,0::2,:,:]
        #s = torch.exp(log_scale_shift[:,1::2,:,:])
        s = torch.sigmoid_(log_scale_shift[:,1::2,:,:] + 2.0)

 
        #Apply the affine transformation
        if not rev:
            x = s*x + t
            log_det_jac = torch.log(s).sum(dim=[1,2,3])
            
        else:
            x = (x-t)/s
            log_det_jac = -torch.log(s).sum(dim=[1,2,3])
            
        return (x,), log_det_jac
        
        
    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims    

class _BaseCouplingBlock(Fm.InvertibleModule):
    '''Base class to implement various coupling schemes.  It takes care of
    checking the dimensions, conditions, clamping mechanism, etc.
    Each child class only has to implement the _coupling1 and _coupling2 methods
    for the left and right coupling operations.
    (In some cases below, forward() is also overridden)
    '''

    def __init__(self, dims_in, dims_c=[],
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN",
                 split_len: Union[float, int] = 0.5):
        '''
        Additional args in docstring of base class.

        Args:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
          split_len: Specify the dimension where the data should be split.
            If given as int, directly indicates the split dimension.
            If given as float, must fulfil 0 <= split_len <= 1 and number of
            unchanged dimensions is set to `round(split_len * dims_in[0, 0])`.
        '''

        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        if isinstance(split_len, float):
            if not (0 <= split_len <= 1):
                raise ValueError(f"Float split_len must be in range [0, 1], "
                                 f"but is: {split_len}")
            split_len = round(self.channels * split_len)
        else:
            if not (0 <= split_len <= self.channels):
                raise ValueError(f"Integer split_len must be in range "
                                 f"0 <= split_len <= {self.channels}, "
                                 f"but is: {split_len}")
        self.split_len1 = split_len
        self.split_len2 = self.channels - split_len

        self.clamp = clamp

        #assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
        #    F"Dimensions of input {dims_in} and one or more conditions {dims_c} don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = (lambda u: 2. * (torch.sigmoid(u) - 0.5))
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations

        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)

        if not rev:
            x2_c = torch.cat([x2, *c], 1) if self.conditional else x2
            y1, j1 = self._coupling1(x1, x2_c)

            y1_c = torch.cat([y1, *c], 1) if self.conditional else y1
            y2, j2 = self._coupling2(x2, y1_c)
        else:
            # names of x and y are swapped for the reverse computation
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            y2, j2 = self._coupling2(x2, x1_c, rev=True)

            y2_c = torch.cat([y2, *c], 1) if self.conditional else y2
            y1, j1 = self._coupling1(x1, y2_c, rev=True)

        return (torch.cat((y1, y2), 1),), j1 + j2

    def _coupling1(self, x1, u2, rev=False):
        '''The first/left coupling operation in a two-sided coupling block.

        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def _coupling2(self, x2, u1, rev=False):
        '''The second/right coupling operation in a two-sided coupling block.

        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j2 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        '''
        raise NotImplementedError()

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims


# Coupling layer which takes conditional info into the subnet (not concatenated)
class AffineCouplingOneSidedContext(_BaseCouplingBlock):
    '''Half of a coupling block following the GLOWCouplingBlock design.  This
    means only one affine transformation on half the inputs.  In the case where
    random permutations or orthogonal transforms are used after every block,
    this is not a restriction and simplifies the design.  '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 clamp: float = 2.,
                 clamp_activation: Union[str, Callable] = "ATAN",
                 split_len: Union[float, int] = 0.5,
                 context_dim=512):
        '''
        Additional args in docstring of base class.

        Args:
          subnet_constructor: function or class, with signature
            constructor(dims_in, dims_out).  The result should be a torch
            nn.Module, that takes dims_in input channels, and dims_out output
            channels. See tutorial for examples. One subnetwork will be
            initialized in the block.
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        '''

        super().__init__(dims_in, dims_c, clamp, clamp_activation,
                         split_len=split_len)
        self.subnet = subnet_constructor(self.split_len1, 2 * self.split_len2, context_dim=context_dim)

    def forward(self, x, c=[], rev=False, jac=True):
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
        #x1_c = torch.cat([x1, *c], 1) if self.conditional else x1

        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        #a = self.subnet(x1_c)
        a = self.subnet(x1, context=c[0])
        s, t = a[:, :self.split_len2], a[:, self.split_len2:]
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t) * torch.exp(-s)
            j *= -1
        else:
            y2 = x2 * torch.exp(s) + t

        return (torch.cat((x1, y2), 1),), j


class iMap(Fm.InvertibleModule):

    def __init__(self, dims_in, dims_c=[], subnet_constructor: Callable = None):

        super().__init__(dims_in, dims_c)

        self.channels, self.h, self.w = dims_in[0]

        # Define the weights and initialize
        self.weight = torch.empty([self.channels, self.channels, 1])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight = torch.nn.Parameter(self.weight)

        # Define the bias and initialize
        self.bias = torch.empty([self.channels])
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.bias = torch.nn.Parameter(self.bias)

        # Define the parameter s
        self.register_parameter('s', nn.Parameter(torch.randn([1,self.channels,1])))
        self.register_parameter('offset', nn.Parameter(torch.ones([1])*8))
        self.pool1 = torch.nn.AvgPool1d(self.channels)

        self.sig = nn.Sigmoid()

        # Get the checkerboard mask
        checkerboard = torch.zeros(self.channels, self.h*self.w)
        checkerboard[1::2, ::2] = 1
        checkerboard[::2, 1::2] = 1
        self.mask = checkerboard


    def forward(self, x, c=[], rev=False, jac=True):

        # x is passed as a list, so use x[0]
        x = x[0]

        self.num_channels = x.shape[-1]**2
        self.mask = self.mask.to(x.device)


        if not rev:
            x_masked = x.view(-1, self.channels, self.h*self.w) * self.mask
            z = torch.nn.functional.conv1d(x_masked, self.weight, bias=self.bias) #1D convolution
            z_new = z.transpose(1,2)
            pool_out = self.pool1(z_new) # Average pooling
            attn_out = (self.sig(pool_out.squeeze(-1)+self.offset)+1e-5).unsqueeze(1)
            attn_mask = (1 - self.mask) * attn_out + self.mask * (self.sig(self.s)+1e-5)
            out = x * attn_mask.view(-1, self.channels, self.h*self.w).view(-1, self.channels, self.h, self.w)
            log_det_jac = torch.sum((self.channels//2) * (torch.log(self.sig(pool_out.squeeze(-1)+self.offset)+1e-5)), dim=-1)
            log_det_jac = log_det_jac + torch.sum(torch.log(self.sig(self.s)+1e-5) * self.mask)

        else:
            s_sig = self.sig(self.s)+1e-5
            s_sig_in = torch.ones_like(s_sig) / s_sig
            x_masked = x.view(-1, self.channels, self.h*self.w) * self.mask * s_sig_in
            out_conv = torch.nn.functional.conv1d(x_masked, self.weight, bias=self.bias) #1D convolution
            pool_out = self.pool1(out_conv.transpose(1,2)) # Average pooling
            attn_out = (self.sig(pool_out.squeeze(2)+self.offset)+1e-5).unsqueeze(1)
            attn_out = torch.ones_like(attn_out)/attn_out
            attn_mask = (1 - self.mask) * attn_out + self.mask * s_sig_in
            out = x * attn_mask.view(-1, self.channels, self.h*self.w).view(-1, self.channels, self.h, self.w)
            log_det_jac = -torch.sum((self.channels//2) * (torch.log(self.sig(pool_out.squeeze(-1)+self.offset)+1e-5)), dim=-1)
            log_det_jac = log_det_jac - torch.sum(torch.log(self.sig(self.s)+1e-5) * self.mask)

        return (out,), log_det_jac

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims




#%%

if __name__=='__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    layer = iMap(dims_in=[[16,320,320]])
    x = [torch.randn([1,16,320,320]).cpu()]

    with torch.no_grad():
        out, jac = layer(x)
        out2, jac2 = layer(out)
        out3, jac3 = layer(out2, rev=True)
        out4, jac4 = layer(out3, rev=True)
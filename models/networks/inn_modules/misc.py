#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
misc.py
    - Modules for downsampling, upsampling, splitting, and permutations in the normalizing flow
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence
import FrEIA.modules as Fm
import warnings
from scipy.stats import special_ortho_group


class Fixed1x1ConvOrthogonal(Fm.InvertibleModule):
    '''Given an invertible matrix M, a 1x1 convolution is performed using M as
    the convolution kernel. Effectively, a matrix muplitplication along the
    channel dimension is performed in each pixel.
    
    The invertible matrix M is computed using scipy.stats.special_ortho_group and the shape is 
    automatically inferred from the data
    '''

    def __init__(self, dims_in, dims_c=None):
        super().__init__(dims_in, dims_c)

        self.channels = dims_in[0][0]
        if self.channels > 512:
            warnings.warn(("scipy.stats.special_ortho_group  will take a very long time to initialize "
                           f"with {self.channels} feature channels."))

        M = special_ortho_group.rvs(self.channels)

        self.M = nn.Parameter(torch.FloatTensor(M).view(*M.shape, 1, 1), requires_grad=False)
        self.M_inv = nn.Parameter(torch.FloatTensor(M.T).view(*M.shape, 1, 1), requires_grad=False)
        #self.logDetM = nn.Parameter(torch.slogdet(M)[1], requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        #n_pixels = x[0][0, 0].numel()
        j = 0.0#self.logDetM * n_pixels
        # log abs det of special ortho matrix is always zero (det is either 1 or -1 )
        if not rev:
            return (F.conv2d(x[0], self.M),), j
        else:
            return (F.conv2d(x[0], self.M_inv),), -j

    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        if len(input_dims[0]) != 3:
            raise ValueError(f"{self.__class__.__name__} requires 3D input (channels, height, width)")
        return input_dims
    
    

    
    
class Split(Fm.InvertibleModule):
    """Invertible split operation.
    Splits the incoming tensor along the given dimension, and returns a list of
    separate output tensors. The inverse is the corresponding merge operation.
    When section size is smaller than dims_in, a new section with the remaining dimensions is created.
    """

    def __init__(self,
                 dims_in: Sequence[Sequence[int]],
                 section_sizes: Union[int, Sequence[int]] = None,
                 n_sections: int = 2,
                 dim: int = 0,
     ):
        """Inits the Split module with the attributes described above and
        checks that split sizes and dimensionality are compatible.
        Args:
          dims_in:
            A list of tuples containing the non-batch dimensionality of all
            incoming tensors. Handled automatically during compute graph setup.
            Split only takes one input tensor.
          section_sizes:
            If set, takes precedence over ``n_sections`` and behaves like the
            argument in torch.split(), except when a list of section sizes is given
            that doesn't add up to the size of ``dim``, an additional split section is
            created to take the slack. Defaults to None.
          n_sections:
            If ``section_sizes`` is None, the tensor is split into ``n_sections``
            parts of equal size or close to it. This mode behaves like
            ``numpy.array_split()``. Defaults to 2, i.e. splitting the data into two
            equal halves.
          dim:
            Index of the dimension along which to split, not counting the batch
            dimension. Defaults to 0, i.e. the channel dimension in structured data.
        """
        super().__init__(dims_in)

        # Size and dimensionality checks
        assert len(dims_in) == 1, "Split layer takes exactly one input tensor"
        assert len(dims_in[0]) >= dim, "Split dimension index out of range"
        self.dim = dim
        l_dim = dims_in[0][dim]

        if section_sizes is None:
            assert 2 <= n_sections, "'n_sections' must be a least 2"
            if l_dim % n_sections != 0:
                warnings.warn('Split will create sections of unequal size')
            self.split_size_or_sections = (
                [l_dim//n_sections + 1] * (l_dim%n_sections) +
                [l_dim//n_sections] * (n_sections - l_dim%n_sections))
        else:
            if isinstance(section_sizes, int):
                assert section_sizes < l_dim, "'section_sizes' too large"
            else:
                assert isinstance(section_sizes, (list, tuple)), \
                    "'section_sizes' must be either int or list/tuple of int"
                assert sum(section_sizes) <= l_dim, "'section_sizes' too large"
                if sum(section_sizes) < l_dim:
                    warnings.warn("'section_sizes' too small, adding additional section")

                    #This line was modified from the FrEIA code
                    section_sizes.append(l_dim - sum(section_sizes))
            self.split_size_or_sections = section_sizes

    def forward(self, x, rev=False, jac=True):
        """See super class InvertibleModule.
        Jacobian log-det of splitting is always zero."""
        if rev:
            return [torch.cat(x, dim=self.dim+1)], 0
        else:
            return torch.split(x[0], self.split_size_or_sections,
                               dim=self.dim+1), 0

    def output_dims(self, input_dims):
        """See super class InvertibleModule."""
        assert len(input_dims) == 1, "Split layer takes exactly one input tensor"
        # Assemble dims of all resulting outputs
        return [tuple(input_dims[0][j] if (j != self.dim) else section_size
                      for j in range(len(input_dims[0])))
                for section_size in self.split_size_or_sections]




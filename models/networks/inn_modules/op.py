#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
op.py
    - Operational modules for the normalizing flow
"""
import FrEIA.modules as Fm

# Perform the squeeze operation
class SqueezeFlow(Fm.InvertibleModule):
    
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)
        
    def forward(self, x, rev=False, jac = True):
        x=x[0]
        b, c, h, w = x.shape
        
        if not rev:
            #Forward operation: h x w x c -> h/2 x w/2 x 4c
            x = x.reshape(b,c, h//2, 2, w//2, 2)
            x = x.permute(0,1,3,5,2,4)
            x = x.reshape(b, 4*c, h//2, w//2)
            
        else:
            #Reverse operation: h/2 x w/2 x 4c -> h x w x c
            x = x.reshape(b,c//4, 2,2,h,w)
            x = x.permute(0,1,4,2,5,3)
            x = x.reshape(b,c//4, h*2, w*2)
            
        return [x,], 0
    
    def output_dims(self, input_dims):
        '''See base class for docstring'''
        #print(input_dims)
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return [(input_dims[0][0] * 4, input_dims[0][1]//2, input_dims[0][2]//2)]
    


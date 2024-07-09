# -*- coding: utf-8 -*-
import torch
import fastmri
from .network_utils import chans_to_coils, format_multicoil

    
class DCLossMulticoil(torch.nn.Module):
    '''
    Data consistency loss for multicoil images
    Used for UNet to only train on the nullspace components
    '''
    
    def __init__(self) -> None:
        super(DCLossMulticoil, self).__init__()
        
    def forward(self, x: torch.Tensor, target:torch.Tensor, mask:torch.Tensor, std:torch.Tensor) -> torch.Tensor:

        #Convert to coils
        x = format_multicoil(x, chans=False)
        target = format_multicoil(target, chans=False)
        
        #Calculate the different images (Note: only need std since mean gets cancelled)
        #diff_img = std.reshape(-1,1,1,1,1)*(target - x)
        diff_img = target - x

        #Find the Fourier transforms of the difference image
        f_diff = fastmri.fft2c(diff_img)
        
        #Invert the mask to only account for unknown Fourier components
        mask = mask*-1 + 1
        
        #Apply the mask (starts as (batch_size, 1, num_rows))
        #mask2D = mask[0].permute(0,2,1).repeat(1,mask.shape[2],1) #Make the mask 2D for removing columns
        mask2D = mask.permute(0, 1, 3, 2).repeat(1, 1, mask.shape[2], 1)
        #f_diff_masked = mask2D.unsqueeze(-1).unsqueeze(0).repeat(1,1,1,1,2) * f_diff
        f_diff_masked = mask2D.unsqueeze(-1).repeat(1, 1, 1, 1, 2) * f_diff
        
        #Calculate the squared error
        loss = torch.mean(torch.square(torch.norm(f_diff_masked, p ='fro')))

        return loss
    

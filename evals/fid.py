#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import os
import torch
import torch.nn as nn
from torchvision.models import vgg16

import torchvision.transforms as transforms
import fastmri
from tqdm import tqdm

import sys
sys.path.append('../')
from util import viz, network_utils, torch_losses, helper
from evals.piq import fid
import variables
from models.pl_models.e2evarnet import VarNetModule

# VGG Embedding
class VGG16Embedding:
    def __init__(self, parallel=False):
        vgg_model = vgg16(weights='IMAGENET1K_V1').eval()
        vgg_model = WrapVGG(vgg_model).cuda()
        if parallel:
            vgg_model = nn.DataParallel(vgg_model)

        self.vgg_model = vgg_model

    def __call__(self, x):
        return self.vgg_model(x)

class WrapVGG(nn.Module):
    def __init__(self, net):
        super(WrapVGG, self).__init__()
        self.features = list(net.features)
        self.features = nn.Sequential(*self.features)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out).view(x.size(0), -1)
        return out
    

def symmetric_matrix_square_root_torch(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    u, s, v = torch.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = s
    si[torch.where(si >= eps)] = torch.sqrt(si[torch.where(si >= eps)])

    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return torch.matmul(torch.matmul(u, torch.diag(si)), v)


def trace_sqrt_product_torch(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root_torch(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))

    return torch.trace(symmetric_matrix_square_root_torch(sqrt_a_sigmav_a))



class CFIDMetric:
    """Helper function for calculating CFID metric.

    Note: This code is adapted from Facebook's FJD implementation in order to compute
    CFID in a streamlined fashion.


    """

    def __init__(self,
                 model,
                 data_loader,
                 data,
                 image_embedding,
                 condition_embedding,
                 resolution,
                 cuda=False,
                 num_samps=1,
                 temp=1.0,
                 rss=False):

        # Necessary args
        self.model = model
        self.loader = data_loader
        self.data = data
        self.image_embedding = image_embedding
        self.condition_embedding = condition_embedding
        self.resolution = resolution
        #self.mri_type = model.config['data_args']['mri_type']

        # Optional args
        self.cuda = cuda
        self.num_samples = num_samps
        self.temp = temp
        self.rss = rss

        # Transform for VGG16
        self.transforms = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def _get_embed_im(self, multicoil_inp):
        # Define new tensor which will be passed to embedding network
        embed_ims = torch.zeros(size=(multicoil_inp.size(0), 3, multicoil_inp.size(-2), multicoil_inp.size(-1)),
                                device='cuda')

        for i in range(multicoil_inp.size(0)):
            single_im = multicoil_inp[i]

            # Normalize image to be in [0, 1]
            im = (single_im - torch.min(single_im)) / (torch.max(single_im) - torch.min(single_im))

            embed_ims[i, :, :, :] = im.repeat(3, 1, 1)

        return embed_ims

    # Get the generated embeddings
    def _get_generated_distribution(self):
        image_embed = []
        cond_embed = []
        true_embed = []

        for i, data in tqdm(enumerate(self.loader),
                            desc='Computing generated distribution',
                            total=len(self.loader)):

            # Get the data
            cond = data[0].to(self.model.device)
            gt = data[1].to(self.model.device)
            mask = data[2].to(self.model.device)
            norm_val = data[3].to(self.model.device)

            # Get the sensitivity maps
            if not self.rss:
                maps = network_utils.get_maps(cond, num_acs=self.model.acs_size, normalizing_val=norm_val)
            else:
                maps = None


            with torch.no_grad():
                if isinstance(self.model, VarNetModule):
                    mask = mask.unsqueeze(1)

                # Get the reconstructions
                samples = self.model.reconstruct(cond,
                                            num_samples = self.num_samples,
                                            temp = self.temp,
                                            check=True,
                                            maps=maps,
                                            mask=mask,
                                            norm_val=norm_val,
                                            split_num=4,
                                            rss=self.rss)

                if isinstance(samples, list):
                    # Stack the samples
                    samples = torch.concat(samples, dim=0)

                # Otherwise, it is the VarNet with a single sample per batch

                # Unnormalize the gt and cond for evaluation
                cond = network_utils.unnormalize(cond, norm_val)
                gt = network_utils.unnormalize(gt, norm_val)

                # Get the magnitude images
                gt = network_utils.get_magnitude(gt, maps=maps, rss=self.rss)
                cond = network_utils.get_magnitude(cond, maps=maps, rss=self.rss)

                # Prepare the images for the embedding network
                image = self._get_embed_im(samples).to(self.model.device)
                condition_im = self._get_embed_im(cond).to(self.model.device)
                true_im = self._get_embed_im(gt).to(self.model.device)

                # Apply normalization transform then pass to VGG
                # Note: You do not have to resize images for VGG in PyTorch
                img_e = self.image_embedding(self.transforms(image))
                cond_e = self.condition_embedding(self.transforms(condition_im))
                true_e = self.image_embedding(self.transforms(true_im))

            true_embed.append(true_e.cpu())
            image_embed.append(img_e.cpu())
            cond_embed.append(cond_e.cpu())
            

        true_embed = torch.cat(true_embed, dim=0)
        image_embed = torch.cat(image_embed, dim=0)
        cond_embed = torch.cat(cond_embed, dim=0)

        # Return double precision tensors
        return image_embed.to(dtype=torch.float64), cond_embed.to(dtype=torch.float64), true_embed.to(dtype=torch.float64)


    def get_cfid(self,y_predict, x_true, y_true):

        # mean estimations
        y_true = y_true.to(x_true.device)
        m_y_predict = torch.mean(y_predict, dim=0)
        m_x_true = torch.mean(x_true, dim=0)
        m_y_true = torch.mean(y_true, dim=0)

        no_m_y_true = y_true - m_y_true
        no_m_y_pred = y_predict - m_y_predict
        no_m_x_true = x_true - m_x_true

        c_y_predict_x_true = torch.matmul(no_m_y_pred.t(), no_m_x_true) / y_predict.shape[0]
        c_y_predict_y_predict = torch.matmul(no_m_y_pred.t(), no_m_y_pred) / y_predict.shape[0]
        c_x_true_y_predict = torch.matmul(no_m_x_true.t(), no_m_y_pred) / y_predict.shape[0]

        c_y_true_x_true = torch.matmul(no_m_y_true.t(), no_m_x_true) / y_predict.shape[0]
        c_x_true_y_true = torch.matmul(no_m_x_true.t(), no_m_y_true) / y_predict.shape[0]
        c_y_true_y_true = torch.matmul(no_m_y_true.t(), no_m_y_true) / y_predict.shape[0]

        inv_c_x_true_x_true = torch.linalg.pinv(torch.matmul(no_m_x_true.t(), no_m_x_true) / y_predict.shape[0])

        c_y_true_given_x_true = c_y_true_y_true - torch.matmul(c_y_true_x_true,
                                                               torch.matmul(inv_c_x_true_x_true, c_x_true_y_true))
        c_y_predict_given_x_true = c_y_predict_y_predict - torch.matmul(c_y_predict_x_true,
                                                                        torch.matmul(inv_c_x_true_x_true,
                                                                                     c_x_true_y_predict))
        c_y_true_x_true_minus_c_y_predict_x_true = c_y_true_x_true - c_y_predict_x_true
        c_x_true_y_true_minus_c_x_true_y_predict = c_x_true_y_true - c_x_true_y_predict

        # Distance between Gaussians
        m_dist = torch.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
        c_dist1 = torch.trace(
            torch.matmul(torch.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_c_x_true_x_true),
                         c_x_true_y_true_minus_c_x_true_y_predict))
        c_dist_2_1 = torch.trace(c_y_true_given_x_true + c_y_predict_given_x_true)
        c_dist_2_2 = - 2 * trace_sqrt_product_torch(
            c_y_predict_given_x_true, c_y_true_given_x_true)

        c_dist2 = c_dist_2_1 + c_dist_2_2

        cfid = m_dist + c_dist1 + c_dist2

        return cfid.cpu().numpy()

    def get_fid(self, y_predict, train_loader):

        # Get the statistics for the datasets
        mu_x, sigma_x = self._get_data_distribution(train_loader)

        # Get the statistics for the generated image
        mu_y, sigma_y = fid._compute_statistics(y_predict.detach().to(dtype=torch.float64))

        score = fid._compute_fid(mu_x, sigma_x, mu_y, sigma_y)

        return score

    def _get_data_distribution(self, loader):

        # Get the directory of the dataset
        #base_dir = variables.fastmri_paths[self.model.config['data_args']['mri_type']]
        base_dir = self.data.base_path

        save_dir_mu = os.path.join(base_dir, 'statsmu.pt')
        save_dir_sigma = os.path.join(base_dir, 'statssigma.pt')
        true_embed = []

        # If the file exists, just open the statistics file
        if os.path.exists(save_dir_mu):
            mu = torch.load(save_dir_mu)
            sigma = torch.load(save_dir_sigma)
            return mu, sigma  # mu and sigma for the dataset

        else:

            for i, data in tqdm(enumerate(loader),
                                desc='Computing data distribution',
                                total=len(loader)):

                with torch.no_grad():

                        # Get the data
                        obs = data[0].to(self.model.device)
                        gt = data[1].to(self.model.device)
                        mask = data[2].to(self.model.device)
                        norm_val = data[3].to(self.model.device)

                        maps = network_utils.get_maps(obs, num_acs=self.model.acs_size, normalizing_val=norm_val)

                        gt = gt.to('cuda')
                        norm_val = norm_val.to('cuda')

                        #Unnormalize the gt
                        gt = network_utils.unnormalize(gt, norm_val)

                        #Get singlecoil
                        gt_mag = network_utils.get_magnitude(gt, maps=maps, rss=self.rss)

                        # Prepare the images for the embedding network
                        true_im = self._get_embed_im(gt_mag).to('cuda')

                        # Apply normalization transform then pass to VGG
                        # Note: You do not have to resize images for VGG in PyTorch
                        true_e = self.image_embedding(self.transforms(true_im))

                        true_embed.append(true_e.cpu())

            true_embed = torch.cat(true_embed, dim=0)
            true_embed = true_embed.to(dtype=torch.float64)

            #Find the metrics
            mu_x, sigma_x = fid._compute_statistics(true_embed.detach().to(dtype=torch.float64))

            #Save the statistics for future use
            torch.save(mu_x, save_dir_mu)
            torch.save(sigma_x, save_dir_sigma)

            return mu_x, sigma_x




        
if __name__ == '__main__':
    x = torch.randn(10, 3, 320, 320).cuda()

    vgg = VGG16Embedding()

    output = vgg(x)
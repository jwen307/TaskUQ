#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional

import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import os
import yaml
from pathlib import Path
import torch
from tqdm import tqdm
from collections import defaultdict
import fastmri

import sys

sys.path.append('../')
from util import  viz, network_utils, torch_losses, helper
from datasets.masks.mask import get_mask, apply_mask


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def psnr_complex(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR) for the entire volume"""

    gt = torch.from_numpy(gt)
    pred = torch.from_numpy(pred)

    #Use the max of the ground truth magnitude
    if maxval is None:
        gt_mag = fastmri.complex_abs(gt)
        maxval = gt_mag.max()

    #Take the difference in the complex domain, then find the magnitude of the difference
    diff = gt - pred
    diff_mag = fastmri.complex_abs(diff)
    mse = torch.pow(diff_mag, 2).mean()

    #Find the PSNR
    psnr = 10 * torch.log10((maxval ** 2) / mse)

    return psnr.numpy()


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim >= 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.shape[-1] == 2:
        is_complex = True
    else:
        is_complex = False

    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        if is_complex:
            #Note: the channel axis should correspond to the dimension with real and imaginary
            ssim = ssim + structural_similarity(
                gt[slice_num], pred[slice_num], channel_axis=-1, data_range=maxval
            )
        else:
            ssim = ssim + structural_similarity(
                gt[slice_num], pred[slice_num], data_range=maxval
            )

    return ssim / gt.shape[0]


def rsnr(gt: np.ndarray, pred: np.ndarray):
    # Do RSNR not in dB (convert to dB after finding the mean)
    # rsnr = np.linalg.norm(gt)**2 / np.linalg.norm(gt - pred)**2

    # Do it in dB
    rsnr = 10 * np.log10(np.linalg.norm(gt) ** 2 / np.linalg.norm(gt - pred) ** 2)

    return rsnr


def calc_metrics(reconstructions, targets, is_complex=False):
    nmses = []
    psnrs = []
    ssims = []
    rsnrs = []
    for fname in tqdm(reconstructions):
        nmses.append(nmse(targets[fname], reconstructions[fname]))
        if is_complex:
            psnrs.append(psnr_complex(targets[fname], reconstructions[fname]))
        else:
            psnrs.append(psnr(targets[fname], reconstructions[fname]))
        ssims.append(ssim(targets[fname], reconstructions[fname]))
        rsnrs.append(rsnr(targets[fname], reconstructions[fname]))

    # report_dict = {
    #     'results': {
    #         'mean_nmse': float(np.mean(nmses)),
    #         'std_err_nmse': float(np.std(nmses) / np.sqrt(len(nmses))),
    #         'mean_psnr': float(np.mean(psnrs)),
    #         'std_err_psnr': float(np.std(psnrs) / np.sqrt(len(psnrs))),
    #         'mean_ssim': float(np.mean(ssims)),
    #         'std_err_ssim': float(np.std(ssims) / np.sqrt(len(ssims))),
    #         'mean_rsnr (db)': float(np.mean(rsnrs)),
    #         'std_err_rsnr (db)': float(np.std(rsnrs) / np.sqrt(len(rsnrs))),
    #     }
    # }
    report_dict = {
        'results': {
            'mean_nmse': f'{float(np.mean(nmses)):.5f} +/- {float(np.std(nmses) / np.sqrt(len(nmses))):.5f}',
            'mean_rsnr (db)': f'{float(np.mean(rsnrs)):.5f} +/- {float(np.std(rsnrs) / np.sqrt(len(rsnrs))):.5f}',
            'mean_psnr': f'{float(np.mean(psnrs)):.5f} +/- {float(np.std(psnrs) / np.sqrt(len(psnrs))):.5f}',
            'mean_ssim': f'{float(np.mean(ssims)):.5f} +/- {float(np.std(ssims) / np.sqrt(len(ssims))):.5f}',
        }
    }

    return report_dict


# %%
# Evaluate the model      
def evaluate(model, model_path, dataset, num_samples=100, model_type='cinn'):
    # if __name__ == '__main__':
    model = model.cuda()
    model = model.eval()

    # Get the path to save the metric scores
    report_path = os.path.join(model_path, 'benchmarks')
    Path(report_path).mkdir(parents=True, exist_ok=True)

    num_val_images = len(dataset.val)

    # Dictionaries for the different types of images
    reconstructions_pd = defaultdict(list)
    targets_pd = defaultdict(list)
    reconstructions_pdfs = defaultdict(list)
    targets_pdfs = defaultdict(list)

    # Get the path to save the reconstruction dictionaries
    recon_path = os.path.join(model_path, 'recons')
    Path(recon_path).mkdir(parents=True, exist_ok=True)

    # Check if the reconstructions have already been found
    if not os.path.exists(os.path.join(recon_path, 'reconstructions_pdfs.yaml')):

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataset.val_dataloader())):

                obs, gt, mask, norm_val, acquisition, fname, slice_num = batch

                # Get the maps for multicoil
                # if challenge == 'multicoil':
                #     maps = network_utils.get_maps(obs, num_acs=model.acs_size, normalizing_val=std)


                obs = obs.to(model.device)
                mask = mask.to(model.device)
                norm_val = norm_val.to(model.device)

                # Get the magnitude image
                target = fastmri.rss_complex(
                    network_utils.format_multicoil(network_utils.unnormalize(gt, norm_val), chans=False), dim=1)

                if model_type == 'VarNet':

                    # Get the masked k-space from the zero_filled image
                    masked_kspace = fastmri.fft2c(
                        network_utils.format_multicoil(network_utils.unnormalize(obs, norm_val), chans=False))
                    # Re-zero out
                    masked_kspace = apply_mask(masked_kspace, mask.unsqueeze(1))



                    # Get the reconstruction
                    output = model(masked_kspace, mask.unsqueeze(1).to(torch.bool), num_low_frequencies=None)
                    reco = output.cpu().float().numpy()

                else:
                    samples = model.reconstruct(obs,
                                                num_samples,
                                                temp=1.0,
                                                check=True,
                                                maps=None,
                                                mask=mask,
                                                norm_val=norm_val,
                                                split_num=4,
                                                multicoil=False,
                                                rss=True)

                    mean_samples = []

                    for b in range(len(samples)):
                        # Get all the posterior samples for a single zf image
                        posteriors = samples[b]

                        # Find the mean sample
                        mean_samples.append(posteriors.mean(dim=0))


                    reco = torch.stack(mean_samples, dim=0).cpu()
                    reco = reco.squeeze(1).numpy()

                if len(reco.shape) != len(target.shape):
                    print('Target shape: ', target.shape)
                    print('Reco shape: ', reco.shape)
                    raise(ValueError('Reconstruction and target shapes do not match'))


                # Get the samples for the volume calculations
                for k in range(gt.shape[0]):

                    if acquisition[k] == 'CORPDFS_FBK':
                        reconstructions_pdfs[fname[k]].append((int(slice_num[k]), reco[k]))
                        targets_pdfs[fname[k]].append((int(slice_num[k]), target[k]))
                    else:
                        reconstructions_pd[fname[k]].append((int(slice_num[k]), reco[k]))
                        targets_pd[fname[k]].append((int(slice_num[k]), target[k]))

        # Save dictionaries as pickles
        helper.write_pickle(reconstructions_pdfs, os.path.join(recon_path, 'reconstructions_pdfs.yaml'))
        # util.write_pickle(targets_pdfs, os.path.join(recon_path, 'targets_pdfs.yaml'))
        helper.write_pickle(reconstructions_pd, os.path.join(recon_path, 'reconstructions_pd.yaml'))
        # util.write_pickle(targets_pd, os.path.join(recon_path, 'targets_pd.yaml'))

    # Load the dictionaries if they already exist
    else:
        print('Loading Reconstructions')
        reconstructions_pdfs = helper.read_pickle(os.path.join(recon_path, 'reconstructions_pdfs.yaml'))
        reconstructions_pd = helper.read_pickle(os.path.join(recon_path, 'reconstructions_pd.yaml'))

        print('Getting Ground Truths')
        # Get the ground truth images
        for i, batch in enumerate(tqdm(dataset.val_dataloader())):

            obs, gt, masks, norm_val, acquisition, fname, slice_num = batch

            # Get the magnitude image
            target = fastmri.rss_complex(
                network_utils.format_multicoil(network_utils.unnormalize(gt, norm_val), chans=False), dim=1)

            # Get the samples for the volume calculations
            for k in range(gt.shape[0]):

                if acquisition[k] == 'CORPDFS_FBK':
                    # reconstructions_pdfs[fname[k]].append((int(slice_num[k]), reco[k]))
                    targets_pdfs[fname[k]].append((int(slice_num[k]), target[k]))
                else:
                    # reconstructions_pd[fname[k]].append((int(slice_num[k]), reco[k]))
                    targets_pd[fname[k]].append((int(slice_num[k]), target[k]))

    # Organize the reconstructiosn by slice for each volume
    # Note: these will all be (num_slices, h, w, 2) and are all numpy arrays
    for fname in reconstructions_pd:
        reconstructions_pd[fname] = np.stack([out for _, out in sorted(reconstructions_pd[fname])])
        targets_pd[fname] = np.stack([out for _, out in sorted(targets_pd[fname])])

        if reconstructions_pd[fname].shape[1] == 2:
            reconstructions_pd[fname] = np.transpose(reconstructions_pd[fname], (0, 2, 3, 1))

    for fname in reconstructions_pdfs:
        reconstructions_pdfs[fname] = np.stack([out for _, out in sorted(reconstructions_pdfs[fname])])
        targets_pdfs[fname] = np.stack([out for _, out in sorted(targets_pdfs[fname])])

        if reconstructions_pdfs[fname].shape[1] == 2:
            reconstructions_pdfs[fname] = np.transpose(reconstructions_pdfs[fname], (0, 2, 3, 1))

    report_dict = {'settings': {'num_val_images': num_val_images,
                                'samples_per_reco': num_samples},
                   }

    # Calculate the magnitude metrics
    print('Calculating the magnitude metrics')
    report_pd_mag = calc_metrics(reconstructions_pd, targets_pd, is_complex=False)
    report_pdfs_mag = calc_metrics(reconstructions_pdfs, targets_pdfs, is_complex=False)
    # Add to the dictionary
    report_dict['results_PD_mag'] = report_pd_mag['results']
    report_dict['results_PDFS_mag'] = report_pdfs_mag['results']

    report_file_path = os.path.join(report_path, 'vol_report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)

    print(report_dict)



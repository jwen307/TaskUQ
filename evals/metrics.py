#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
from tqdm import tqdm
import fastmri

import sys

sys.path.append('../../mri_cnf1/')
from util import network_utils
from evals import fid
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

    gt = network_utils.check_type(gt, 'tensor')
    pred = network_utils.check_type(pred, 'tensor')

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

def psnr_complex_multi(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None, maps=None, rss=True
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR) for the entire volume"""

    gt = network_utils.check_type(gt, 'tensor')
    pred = network_utils.check_type(pred, 'tensor')

    #Use the max of the ground truth magnitude
    if maxval is None:
        #gt_mag = fastmri.complex_abs(gt)
        gt_mag = network_utils.get_magnitude(gt, maps=maps, rss=rss)
        maxval = gt_mag.max()

    #Take the difference in the complex domain, then find the magnitude of the difference
    diff = gt - pred
    diff_mag = network_utils.get_magnitude(diff, maps=maps, rss=rss)
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
                gt[slice_num], pred[slice_num], channel_axis=0, data_range=maxval
            )

    return ssim / gt.shape[0]


def rsnr(gt: np.ndarray, pred: np.ndarray):
    # Compute the reconstruction signal to noise ratio
    rsnr = 10 * np.log10(np.linalg.norm(gt) ** 2 / np.linalg.norm(gt - pred) ** 2)

    return rsnr


def calc_metrics(reconstructions, targets, is_complex=False):
    '''
    Calculate the metrics for a dictionary of reconstructions and targets
    :param reconstructions:
    :param targets:
    :param is_complex: Should you compute PSNR and SSIM for the complex case
    :return:
    '''
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

    report_dict = {
        'results': {
            'mean_nmse': float(np.mean(nmses)),
            'std_err_nmse': float(np.std(nmses) / np.sqrt(len(nmses))),
            'mean_psnr': float(np.mean(psnrs)),
            'std_err_psnr': float(np.std(psnrs) / np.sqrt(len(psnrs))),
            'mean_ssim': float(np.mean(ssims)),
            'std_err_ssim': float(np.std(ssims) / np.sqrt(len(ssims))),
            'mean_rsnr (db)': float(np.mean(rsnrs)),
            'std_err_rsnr (db)': float(np.std(rsnrs) / np.sqrt(len(rsnrs))),
        }
    }

    return report_dict

def calc_metrics_list(reconstructions, targets, is_complex=False):
    '''
    Calculate the metrics for a list of reconstructions and targets
    :param reconstructions:
    :param targets:
    :param is_complex: Should you compute PSNR and SSIM for the complex case
    :return:
    '''
    nmses = []
    psnrs = []
    ssims = []
    rsnrs = []
    for i in tqdm(range(len(reconstructions))):

        if is_complex:
            target = network_utils.format_multicoil(targets[i], chans=False).numpy()
            reconstruction = network_utils.format_multicoil(reconstructions[i], chans=False).numpy()
            psnrs.append(psnr_complex(target, reconstruction))
        else:
            target = targets[i]
            reconstruction = reconstructions[i]
            psnrs.append(psnr(target, reconstruction))

        nmses.append(nmse(target, reconstruction))
        ssims.append(ssim(target, reconstruction))
        rsnrs.append(rsnr(target, reconstruction))

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

    mean_nmse= float(np.mean(nmses))
    std_err_nmse= float(np.std(nmses) / np.sqrt(len(nmses)))
    mean_psnr= float(np.mean(psnrs))
    std_err_psnr= float(np.std(psnrs) / np.sqrt(len(psnrs)))
    mean_ssim =float(np.mean(ssims))
    std_err_ssim= float(np.std(ssims) / np.sqrt(len(ssims)))
    mean_rsnr=  float(np.mean(rsnrs))
    std_err_rsnr=  float(np.std(rsnrs) / np.sqrt(len(rsnrs)))


    report_dict = {
        'mean_results':{
            'NMSE': f'{mean_nmse:.5f} +/- {std_err_nmse:.5f}',
            'RSNR': f'{mean_rsnr:.5f} +/- {std_err_rsnr:.5f}',
            'PSNR': f'{mean_psnr:.5f} +/- {std_err_psnr:.5f}',
            'SSIM': f'{mean_ssim:.5f} +/- {std_err_ssim:.5f}',

        }
    }


    return report_dict


# Get the evaluation metrics for the model
def evaluate_mean(model, data, num_samples, temp, complex=False, rss=False):
    # Get the 72 test images
    dataset = data.test

    # Put the model in eval mode
    model = model.eval()
    model = model.cuda()

    # Get the reconstructions and targets
    preds = []
    gts = []

    for k in tqdm(range(len(dataset))):
        # Get the data
        cond = dataset[k][0].unsqueeze(0).to(model.device)
        gt = dataset[k][1].unsqueeze(0).to(model.device)
        mask = dataset[k][2].unsqueeze(0).to(model.device)
        norm_val = dataset[k][3].unsqueeze(0).to(model.device)

        # Get the sensitivity maps
        if not rss:
            maps = network_utils.get_maps(cond, num_acs=model.acs_size, normalizing_val=norm_val)
        else:
            maps = None

        # Get the reconstruction
        with torch.no_grad():
            samples = model.reconstruct(cond,
                                        num_samples,
                                        temp,
                                        check=True,
                                        maps=maps,
                                        mask=mask,
                                        norm_val=norm_val,
                                        split_num=4,
                                        multicoil=complex,
                                        rss=rss)


        gt = network_utils.unnormalize(gt, norm_val)

        # print(gt.shape)
        # print(samples[0].shape)

        if not complex:
            # Get the singlecoil magnitude images
            gts.append(network_utils.get_magnitude(gt, maps=maps, rss=rss).cpu().numpy())
            mean_pred = samples[0].mean(dim=0).unsqueeze(0)
            preds.append(mean_pred.cpu().numpy())

            # print(gts[-1].shape)
            # print(preds[-1].shape)


        # Get the metrics with the complex images
        else:
            gts.append(network_utils.multicoil2single(gt, maps=maps, rss=rss).cpu().numpy())
            mean_pred = network_utils.multicoil2single(samples[0],maps=maps, rss=rss).mean(dim=0).unsqueeze(0)
            preds.append(mean_pred.cpu().numpy())

    # Calculate the metrics
    metrics_avg = calc_metrics_list(preds, gts, is_complex=complex)
    print(metrics_avg)

    return metrics_avg


# Get the evaluation metrics for the posteriors
def evaluate_posterior(model, data, num_samples, temp, rss=False, test=True, full_dataset=False, get_cfid=True):

    # Use the VGG16 model for embedding
    vgg16 = fid.VGG16Embedding()

    if test:
        loader = data.test_dataloader()
    elif not test and not full_dataset:
        if data.mri_type == 'knee':
            loader = data.val_dataloader()
        #Evaluate on 2376 images for brain dataset
        else:
            idxs = torch.randperm(len(data.val)).numpy()
            eval_idxs = idxs[:2376]
            eval_data = torch.utils.data.Subset(data.val, eval_idxs)
            loader = torch.utils.data.DataLoader(eval_data, batch_size=4, shuffle=True)

    else:
        #Combine the train and val datasets
        eval_data = torch.utils.data.ConcatDataset([data.train, data.val])
        if data.mri_type == 'brain':
            idxs = torch.randperm(len(eval_data)).numpy()
            eval_idxs = idxs[:14576]
            eval_data = torch.utils.data.Subset(eval_data, eval_idxs)
        loader = torch.utils.data.DataLoader(eval_data, batch_size=4, shuffle=True)


    cfid_metric = fid.CFIDMetric(model,
                                 loader,
                                 data,
                                 vgg16,
                                 vgg16,
                                 resolution=data.img_size,
                                 cuda=True,
                                 num_samps=num_samples,
                                 temp=temp,
                                 rss=rss,
                                 )

    # Calculate the embeddings and metric
    y_predict, x_true, y_true = cfid_metric._get_generated_distribution()
    x_true_inter = torch.repeat_interleave(x_true, num_samples, dim=0)
    y_true_inter = torch.repeat_interleave(y_true, num_samples, dim=0)
    if get_cfid:
        cfid_val = cfid_metric.get_cfid(y_predict, x_true_inter, y_true_inter)
    else:
        cfid_val = 0
    fid_val = cfid_metric.get_fid(y_predict, data.train_dataloader())

    report_dict = {
        'posterior_results':{
            'CFID': f'{cfid_val:.5f}',
            'FID': f'{fid_val:.5f}',
    }}

    print(report_dict)

    return report_dict



# Get the evaluation metrics for the model
def evaluate_autoencoder(model, data, complex=False, rss=False, dc=False, sample_posterior=True):
    # Get the 72 test images
    dataset = data.test

    # Put the model in eval mode
    model = model.eval()
    model = model.cuda()

    # Get the reconstructions and targets
    preds = []
    gts = []

    for k in tqdm(range(len(dataset))):
        # Get the data
        cond = dataset[k][0].unsqueeze(0).to(model.device)
        gt = dataset[k][1].unsqueeze(0).to(model.device)
        mask = dataset[k][2].unsqueeze(0).to(model.device)
        norm_val = dataset[k][3].unsqueeze(0).to(model.device)

        # Get the sensitivity maps
        maps = network_utils.get_maps(cond, num_acs=model.acs_size, normalizing_val=norm_val)

        # Get the reconstruction
        with torch.no_grad():
            samples, _ = model(gt,
                            cond=cond,
                            mask=mask,
                            norm_val=norm_val,
                            dc=dc,
                            sample_posterior=sample_posterior,
                            )


        gt = network_utils.unnormalize(gt, norm_val)
        samples = network_utils.unnormalize(samples, norm_val)

        if not complex:
            # Get the singlecoil magnitude images
            gts.append(network_utils.get_magnitude(gt, maps=maps, rss=rss).cpu().numpy())
            preds.append(network_utils.get_magnitude(samples, maps=maps, rss=rss).cpu().numpy())
            #mean_pred = samples[0].mean(dim=0).unsqueeze(0)
            #preds.append(mean_pred.cpu().numpy())


        # Get the metrics with the complex images
        else:
            #gts.append(network_utils.multicoil2single(gt, maps=maps, rss=rss).cpu().numpy().unsqueeze(0))
            #mean_pred = network_utils.multicoil2single(samples[0],maps=maps, rss=rss).mean(dim=0).unsqueeze(0)
            #preds.append(mean_pred.cpu().numpy())
            gts.append(gt.cpu().numpy())
            preds.append(samples.cpu().numpy())

    # Calculate the metrics
    metrics_avg = calc_metrics_list(preds, gts, is_complex=complex)
    print(metrics_avg)

    return metrics_avg


# Get the evaluation metrics for the model
def evaluate_varnet(model, data, complex=False):
    # Get the 72 test images
    dataset = data.test

    # Put the model in eval mode
    model = model.eval()
    model = model.cuda()

    # Get the reconstructions and targets
    preds = []
    gts = []

    for k in tqdm(range(len(dataset))):
        # Get the data
        cond = dataset[k][0].unsqueeze(0).to(model.device)
        gt = dataset[k][1].unsqueeze(0).to(model.device)
        mask = dataset[k][2].unsqueeze(0).unsqueeze(1).to(model.device)
        norm_val = dataset[k][3].unsqueeze(0).to(model.device)

        # Get the sensitivity maps
        #maps = network_utils.get_maps(cond, num_acs=data.acs_size, normalizing_val=norm_val)

        # Get the reconstruction
        with torch.no_grad():
            # Get the masked k-space from the zero_filled image
            masked_kspace = fastmri.fft2c(
                network_utils.format_multicoil(network_utils.unnormalize(cond, norm_val), chans=False))
            # Re-zero out
            masked_kspace = apply_mask(masked_kspace, mask)

            # Get the magnitude image
            target = fastmri.rss_complex(
                network_utils.format_multicoil(network_utils.unnormalize(gt, norm_val), chans=False), dim=1)

            # output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
            output = model(masked_kspace, mask.to(torch.bool), num_low_frequencies=None)


        gts.append(target.cpu().numpy())
        preds.append(output.cpu().numpy())

    # Calculate the metrics
    metrics_avg = calc_metrics_list(preds, gts, is_complex=complex)
    print(metrics_avg)

    return metrics_avg


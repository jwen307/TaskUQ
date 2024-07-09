#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import math
import numpy as np
import torch
import os
import fastmri
from tqdm import tqdm

import sys

sys.path.append("../")


from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper
import variables
from util import network_utils, viz
import matplotlib.pyplot as plt
import conformal_utils as conformal

# Get the input arguments
args = helper.flags()

# Get the classifier network directory
load_ckpt_dir = args.load_ckpt_dir
load_last_ckpt = args.load_last_ckpt

model_folder = os.path.basename(os.path.normpath(load_ckpt_dir))
eval_dir = os.path.join('../logs/Classifier/', model_folder)

# Get the reconstruction network directory
load_recon_dir = args.load_recon_dir
load_last_recon_ckpt = args.load_last_recon_ckpt


num_samples = 32  # Number of posteriors to sample for each slice during calibration
alpha = 0.01  # User set error rate
interval_threshold = 0.1  # Threshold for the interval size before data collection stops
conformal_method = 'cqr'
num_test_posteriors = 32  # Number of posteriors to sample for each slice during testing

print(torch.cuda.is_available())

# %% Get the arguments for training
if __name__ == "__main__":

    # Load the previous configurations
    ckpt_name = 'last.ckpt' if load_last_ckpt else 'best_val_loss.ckpt'
    ckpt = os.path.join(load_ckpt_dir,
                        'checkpoints',
                        ckpt_name)

    # Get the configuration file
    config_file = os.path.join(load_ckpt_dir, 'configs.pkl')
    config = helper.read_pickle(config_file)

    # Load the configuration for the CNF
    cnf_ckpt_name = 'last.ckpt' if load_last_recon_ckpt else 'best.ckpt'
    cnf_ckpt = os.path.join(load_recon_dir,
                            'checkpoints',
                            cnf_ckpt_name)

    # Get the configuration file for the CNF
    cnf_config_file = os.path.join(load_recon_dir, 'configs.pkl')
    cnf_config = helper.read_pickle(cnf_config_file)
    cnf_config['data_args']['specific_label'] = config['data_args']['specific_label']

    # Get the directory of the dataset
    base_dir = variables.fastmri_paths[cnf_config['data_args']['mri_type']]

    # Get the model type
    model_type = 'Classifier'
    recon_type = load_recon_dir.split('/')[-3]

    # Load the models
    model = helper.load_model(model_type, config, ckpt)
    recon_model = helper.load_model(recon_type, cnf_config, cnf_ckpt)

    # %% Evaluate the model
    model.eval()
    model.cuda()
    model.complex = False
    model.challenge = 'singlecoil'

    recon_model.eval()
    recon_model.cuda()
    recon_model.temp = 1.0


    ys = []
    qhats = []


    # First 4 have meniscus tears, last 4 do not
    eval_vols = ['file1000190.h5', 'file1000263.h5', 'file1000264.h5', 'file1000283.h5',
                 'file1000052.h5', 'file1000073.h5', 'file1000182.h5', 'file1000243.h5']



    # %% Calibrate for all the accelerations
    for accel in [16, 8, 4, 2]:
        print('Acceleration: ', accel)

        #%% Find the classifier predictions for the validation set and save it to a file
        save_dir = os.path.join(load_ckpt_dir, load_recon_dir.split('/')[-3], load_recon_dir.split('/')[-2], str(accel), 'clinical_study')

        # Create the directory if it does not already exist and preprocess the files
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

            recon_preds = []
            gt_preds = []

            data = FastMRIDataModule(base_dir,
                                     batch_size=cnf_config['train_args']['batch_size'],
                                     num_data_loader_workers=4,
                                     evaluating=True,
                                     specific_accel=accel,
                                     **cnf_config['data_args'],
                                     )
            data.prepare_data()
            data.setup()

            with torch.no_grad():
                print('Getting classifier predictions...')
                for i in tqdm(range(len(data.val))):
                    c = data.val[i][0].unsqueeze(0).to(model.device)
                    x = data.val[i][1].to(model.device)
                    if recon_type == 'VarNet':
                        masks = data.val[i][2].unsqueeze(0).unsqueeze(1).to(model.device)
                    else:
                        masks = data.val[i][2].to(model.device)
                    norm_val = data.val[i][3].unsqueeze(0).to(model.device)
                    y = data.val[i][-1]

                    # Skip the eval volumes
                    if data.val[i][5] in eval_vols:
                        continue

                    # Get the reconstructions
                    samples = recon_model.reconstruct(c,
                                              num_samples=num_samples,
                                              temp=1.0,
                                              check=True,
                                              maps=None,
                                              mask=masks,
                                              norm_val=norm_val,
                                              split_num=16,
                                              multicoil=False,
                                              rss=True)
                    recons = samples[0]

                    # Get the classifier prediction for the reconstructions
                    if num_samples > 64:
                        ypred = model(recons[0:64]).cpu()
                        yhat = torch.sigmoid(ypred).cpu()
                        for k in range(1, math.ceil(num_samples / 64)):
                            ypred = model(recons[k * 64:(k + 1) * 64]).cpu()
                            yhat = torch.cat((yhat, torch.sigmoid(ypred).cpu()), dim=0)
                    else:
                        ypred = model(recons)
                        yhat = torch.sigmoid(ypred)

                    # Get the ground truth prediction
                    gt = fastmri.rss_complex(network_utils.format_multicoil(network_utils.unnormalize(x, norm_val), chans=False), dim=1)
                    ypred = model(gt)
                    yhat_gt = torch.sigmoid(ypred)

                    # Log the predictions
                    recon_preds.append(yhat.cpu().numpy())
                    gt_preds.append(yhat_gt.cpu().numpy())


            # Save the predictions
            recon_preds = np.stack(recon_preds, axis=0)
            gt_preds = np.concatenate(gt_preds, axis=0)
            np.save(os.path.join(save_dir, 'recon_preds.npy'), recon_preds)
            np.save(os.path.join(save_dir, 'gt_preds.npy'), gt_preds)

        else:
            print('Loading predictions...')
            recon_preds = np.load(os.path.join(save_dir, 'recon_preds.npy'))
            gt_preds = np.load(os.path.join(save_dir, 'gt_preds.npy'))
            print('Number of Calibration Points: ', len(gt_preds))


        # %% Get all the labels for the validation set
        if len(ys) == 0:
            if not os.path.exists(os.path.join(save_dir, 'ys.npy')):
                data = FastMRIDataModule(base_dir,
                                         batch_size=cnf_config['train_args']['batch_size'],
                                         num_data_loader_workers=4,
                                         evaluating=True,
                                         specific_accel=accel,
                                         **cnf_config['data_args'],
                                         )
                data.prepare_data()
                data.setup()

                print('Getting labels...')
                for i in range(len(data.val)):

                    # Skip the eval volumes
                    if data.val[i][5] in eval_vols:
                        continue

                    ys.append(data.val[i][-1])

                np.save(os.path.join(save_dir, 'ys.npy'), ys)
            else:
                print('Loading labels...')
                ys = np.load(os.path.join(save_dir, 'ys.npy'))

        #%% Get the qhat for the reconstructions
        if recon_type == 'VarNet':
            qhat = conformal.conformal_calibration(recon_preds, gt_preds, torch.randperm(len(gt_preds)).numpy(), alpha, conformal_method='residual')
            qhats.append(qhat)
        else:
            qhat = conformal.conformal_calibration(recon_preds[:,:num_samples], gt_preds, torch.randperm(len(gt_preds)).numpy(), alpha, conformal_method=conformal_method)
            qhats.append(qhat)



    #%% Test out the intervals on test volumes
    import h5py
    from datasets.fastmri_multicoil_preprocess import get_compressed, get_normalizing_val
    from datasets.masks.mask import get_mask, apply_mask
    from fastmri.data.transforms import to_tensor
    import time

    img_size = cnf_config['data_args']['img_size']
    mask_type = cnf_config['data_args']['mask_type']

    method_times = [] # Record the amount of time it takes to compute the conformal interval for each slice
    percent_slices_left = [[],[],[],[]] # Record the percentage of slices left after the interval threshold is met
    center_dist = [[] for _ in range(len(eval_vols))] # Keep track of the distances between GT value of a volume and the center of the interval

    coverages = [] # Keep track of the empirical coverage for each volume
    center_coverages = [] # Keep track of the empirical coverage for each volume when the center distance is less than half the threshold

    all_slices_count = 0
    slice_accels = [] #Keep track of the acceleration for each of the slices to take average later


    # Repeat for all the eval volumes
    with torch.no_grad():
        for v, vol in enumerate(eval_vols):
            print('Volume: ', vol)

            num_in_interval = 0  # Keep track of the number of slices in the interval for each volume
            num_in_interval_center = 0 # Keep track of the number of slices with center distance less than half the threshold

            # Load the data
            with h5py.File(os.path.join(base_dir, 'multicoil_val', vol), 'r') as hf:
                # Get the kspace
                kspace = hf['kspace']

                # Get the kspace as a tensor
                kspace_torch = to_tensor(kspace[:])

                # Get the virtual coils
                compressed_k, vhs = get_compressed(kspace_torch, img_size, cnf_config['data_args']['num_vcoils'])

                # Only include the center 0.8 of the slices
                compressed_k = compressed_k[int(0.1 * compressed_k.shape[0]):int(0.9 * compressed_k.shape[0])]

                total_num_slices = len(compressed_k)
                all_slices_count += total_num_slices

                # 1) Get the normalizing value
                masked_kspace, norm_val = get_normalizing_val(compressed_k, img_size,
                                                              mask_type, accel_rate=16)


                # Repeat until the interval threshold is met
                for i, accel in enumerate([16,8,4,2]):
                    print('Acceleration: ', accel)

                    # Start timing here
                    start = time.time()

                    # Get the mask
                    mask = get_mask(accel=accel, size=img_size, mask_type=mask_type)

                    # Apply the mask
                    masked_kspace = apply_mask(compressed_k, mask)

                    # Get the zf imgs
                    masked_imgs = fastmri.ifft2c(masked_kspace)
                    masked_imgs = network_utils.format_multicoil(masked_imgs, chans=True)

                    # 2) Normalize the images of the volume
                    masked_imgs = masked_imgs/ norm_val
                    masked_imgs = masked_imgs.to(recon_model.device)

                    if recon_type == 'VarNet':
                        mask = mask.unsqueeze(0).unsqueeze(1).repeat(masked_imgs.shape[0], 1, 1, 1, 1).to(recon_model.device)
                    else:
                        mask = mask.repeat(masked_imgs.shape[0], 1, 1, 1).to(recon_model.device)


                    # 3) Get the reconstructions
                    samples = recon_model.reconstruct(masked_imgs,
                                                      num_samples=num_test_posteriors,
                                                      temp=1.0,
                                                      check=True,
                                                      maps=None,
                                                      mask=mask,
                                                      norm_val=torch.tensor(norm_val).float().repeat(masked_imgs.shape[0], 1),
                                                      split_num=16,
                                                      multicoil=False,
                                                      rss=True)

                    # Get the classifier prediction for the reconstructions
                    y_hats = []
                    # 4) Pass in the reconstructions of each slice separately
                    if recon_type == 'VarNet':
                        ypred = model(samples.unsqueeze(1))
                        y_hats = torch.sigmoid(ypred).cpu().numpy()

                    else:
                        for j in range(masked_imgs.shape[0]):
                            recons = samples[j]

                            # Get the classifier prediction for the reconstructions
                            if num_test_posteriors > 64:
                                ypred = model(recons[0:64]).cpu()
                                yhat = torch.sigmoid(ypred).cpu()
                                for k in range(1, math.ceil(num_samples / 64)):
                                    ypred = model(recons[k * 64:(k + 1) * 64]).cpu()
                                    yhat = torch.cat((yhat, torch.sigmoid(ypred).cpu()), dim=0)
                            else:
                                ypred = model(recons)
                                yhat = torch.sigmoid(ypred)

                            y_hats.append(yhat.cpu().numpy())

                        y_hats = np.stack(y_hats, axis=0)


                    # 5) Get the interval size
                    if recon_type == 'VarNet':
                        lower_interval, upper_interval, interval_size = conformal.conformal_inference(y_hats, qhats[i], alpha=alpha, conformal_method='residual')
                        print('Max Interval size: ', interval_size.max())
                    else:
                        lower_interval, upper_interval, interval_size = conformal.conformal_inference(y_hats, qhats[i], alpha=alpha, conformal_method=conformal_method)

                        # Print the largest interval size
                        print('Max Interval size: ', interval_size.max())

                    # Record the conformal interval acquisition time
                    method_times.append((time.time()-start)/len(compressed_k))


                    # Check the number of slices in the interval when the threshold is met
                    # GT
                    # See how far the GTs are from the center of the interval
                    leq_thresh = interval_size <= interval_threshold
                    if any(leq_thresh):
                        gt_imgs = fastmri.ifft2c(compressed_k[leq_thresh])
                        gt_imgs = fastmri.rss_complex(gt_imgs, dim=1).unsqueeze(1)

                        # Pass GT through the classifier
                        y_pred_gt = model(gt_imgs)
                        yhat_gt = torch.sigmoid(y_pred_gt)

                        # Find the distance from the center of the interval
                        center_interval = (lower_interval[leq_thresh] + upper_interval[leq_thresh]) / 2
                        distance = np.abs(center_interval - yhat_gt.detach().cpu().numpy())
                        center_dist[v] = np.concatenate([center_dist[v], distance])
                        #print(center_dist)


                        # See how many gt slices are in the intervals
                        lowers = lower_interval[leq_thresh]
                        uppers = upper_interval[leq_thresh]
                        for k in range(len(lowers)):
                            if yhat_gt[k] >= lowers[k] and yhat_gt[k] <= uppers[k]:
                                num_in_interval += 1

                            # Consider cases where the center distance is less than half the threshold as in the interval
                            if distance[k] <= interval_threshold/2:
                                num_in_interval_center += 1

                            # Record the acceleration for each slice
                            slice_accels.append(accel)


                    # 6) Check which interval sizes are less than the threshold
                    # Keep only the slices that are still above the threshold
                    compressed_k = compressed_k[interval_size > interval_threshold]

                    # Record the percentage of slices left
                    percent_slices_left[i].append(len(compressed_k)/total_num_slices)

                    # Break when there are no more slices left
                    if len(compressed_k) == 0:
                        break

                # Include the fully sampled images if threshold never went below
                if len(compressed_k) > 0:
                    num_in_interval += len(compressed_k)
                    num_in_interval_center += len(compressed_k)

                    # Record the acceleration for each slice
                    for _ in range(len(compressed_k)):
                        slice_accels.append(1)


                        # Record the empirical coverage
            coverages.append(num_in_interval/total_num_slices)
            center_coverages.append(num_in_interval_center/total_num_slices)


    # Print the results
    print('Method Times: {0:.3f} +/- {1:.3f}'.format(np.mean(method_times), np.std(method_times)/np.sqrt(len(method_times))))
    #print('Percent Slices Left: {0}'.format([np.mean(x) for x in percent_slices_left]))
    # if recon_type != 'VarNet':
    #     print('Percent Slices Left (Max): {0}'.format([np.max(x) for x in percent_slices_left]))
    #     print('Percent Slices Left (Min): {0}'.format([np.min(x) for x in percent_slices_left]))
    print('Mean Max Center Distances for each Vol: {0:.3f} +/- {1:.3f}'.format(np.mean([np.max(x) for x in center_dist]), np.std([np.max(x) for x in center_dist])/np.sqrt(len(center_dist))))
    print('Mean Empirical Coverages: {0:.3f} +/- {1:.3f}'.format(np.mean(coverages), np.std(coverages)/np.sqrt(len(coverages))))
    print('Mean Center Coverages: {0:.3f} +/- {1:.3f}'.format(np.mean(center_coverages), np.std(center_coverages)/np.sqrt(len(center_coverages))))
    print('Mean Acceleration for each slice: {0:.3f} '.format(1/np.mean(1/np.array(slice_accels))))













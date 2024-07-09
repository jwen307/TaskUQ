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
import conformal_utils as conformal
from util import network_utils, viz
import matplotlib.pyplot as plt

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

num_samples = 32
alpha = 0.05
calib_split = 0.7
num_trials=10000


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

    ys = []


    #%% Find the classifier predictions for the validation set and save it to a file
    for accel in [16, 8, 4, 2]:
        print('Acceleration: ', accel)

        #%% Find the classifier predictions for the validation set and save it to a file
        save_dir = os.path.join(load_ckpt_dir, load_recon_dir.split('/')[-3], load_recon_dir.split('/')[-2], str(accel))

        # Create the directory if it does not already exist and preprocess the files
        if not os.path.exists(os.path.join(save_dir, 'recon_preds.npy')):
            os.makedirs(save_dir, exist_ok=True)

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


                    # Get the reconstructions
                    samples = recon_model.reconstruct(c,
                                              num_samples=num_samples,
                                              temp=1.0,
                                              check=True,
                                              maps=None,
                                              mask=masks,
                                              norm_val=norm_val,
                                              split_num=4,
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


                    ys.append(data.val[i][-1])

                np.save(os.path.join(save_dir, 'ys.npy'), ys)
            else:
                print('Loading labels...')
                ys = np.load(os.path.join(save_dir, 'ys.npy'))


        #%% Get the empirical coverage
        if recon_type == 'VarNet':
            coverages, theo = conformal.conformal_eval_empirical(recon_preds, gt_preds, ys, calib_split=calib_split, alpha=alpha, num_trials=num_trials,
                                conformal_method='residual', save_output_dir=save_dir)

            max_val = max(max(coverages), max(theo))
            min_val = min(min(coverages), min(theo))

            bins = np.arange(0, 1, 1 / (len(ys) * 0.3))

            plt.figure()
            plt.hist(coverages, bins=bins, alpha=0.5, label='Empirical')

            plt.hist(theo, bins=bins, alpha=0.5, label='Theoretical')

            # Plot a line for the mean of the empirical coverages
            empirical_mean = np.mean(coverages)
            plt.axvline(x=empirical_mean, color='navy', linestyle='--', label='Empirical \n Mean')
            # Add a text for the empirical mean
            plt.text(0.951, plt.ylim()[1] * 0.1, f'{empirical_mean:.4f}', color='navy')
            plt.legend(loc='upper left')
            plt.xlim([0.9, 1.0])

            plt.show()

        else:
            coverages_lwr, theo_lwr = conformal.conformal_eval_empirical(recon_preds[:,:num_samples], gt_preds, ys, calib_split=calib_split, alpha=alpha, num_trials=num_trials,
                                    conformal_method='studentized_residual', save_output_dir=save_dir)
            coverages_cqr, theo_cqr = conformal.conformal_eval_empirical(recon_preds[:,:num_samples], gt_preds, ys, calib_split=calib_split, alpha=alpha, num_trials=num_trials,
                                    conformal_method='cqr', save_output_dir=save_dir)

            # Get the plot for the lwr coverage
            max_val = max(max(coverages_lwr), max(theo_lwr))
            min_val = min(min(coverages_lwr), min(theo_lwr))

            bins = np.arange(0, 1, 1 / (len(ys) * 0.3))

            plt.figure()
            plt.hist(coverages_lwr, bins=bins, alpha=0.5, label='Empirical')
            plt.hist(theo_lwr, bins=bins, alpha=0.5, label='Theoretical')

            # Plot a line for the mean of the empirical coverages
            empirical_mean = np.mean(coverages_lwr)
            plt.axvline(x=empirical_mean, color='navy', linestyle='--', label='Empirical \n Mean')
            # Add a text for the empirical mean
            plt.text(0.951, plt.ylim()[1] * 0.1, f'{empirical_mean:.4f}', color='navy')
            plt.legend(loc='upper left')
            plt.xlim([0.9, 1.0])

            plt.show()

            #Get the plot for the cqr coverage
            max_val = max(max(coverages_cqr), max(theo_cqr))
            min_val = min(min(coverages_cqr), min(theo_cqr))

            bins = np.arange(0, 1, 1 / (len(ys) * 0.3))

            # Plot a histogrom of the coverages
            plt.figure()
            plt.hist(coverages_cqr, bins=bins, alpha=0.5, label='Empirical')
            plt.hist(theo_cqr, bins=bins, alpha=0.5, label='Theoretical')

            # Plot a line for the mean of the empirical coverages
            empirical_mean = np.mean(coverages_cqr)
            plt.axvline(x=empirical_mean, color='navy', linestyle='--', label='Empirical \n Mean')
            # Add a text for the empirical mean
            plt.text(0.951, plt.ylim()[1] * 0.1, f'{empirical_mean:.4f}', color='navy')
            plt.legend(loc='upper left')

            plt.xlim([0.9, 1.0])
            plt.show()
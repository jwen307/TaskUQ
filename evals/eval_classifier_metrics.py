#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import math
import numpy as np
import torch
import os
import traceback
import fastmri
import yaml
from tqdm import tqdm
from sklearn.model_selection import KFold

import sys
sys.path.append("../")

from datasets.fastmri_annotated import FastMRIAnnotated
from util import helper
import variables
from util import network_utils
from util.uncertainty import  uncertainty_eval

#Get the input arguments
args = helper.flags()

#Get the checkpoint arguments if needed
load_ckpt_dir = args.load_ckpt_dir
load_last_ckpt = args.load_last_ckpt

model_folder = os.path.basename(os.path.normpath(load_ckpt_dir))
eval_dir = os.path.join('../logs/Classifier/', model_folder)


print(torch.cuda.is_available())

        
#%% Get the arguments for training
if __name__ == "__main__":

    # Load the previous configurations
    ckpt_name = 'last.ckpt' if load_last_ckpt else 'best_val_loss.ckpt'
    ckpt = os.path.join(load_ckpt_dir,
                        'checkpoints',
                        ckpt_name)

    # Get the configuration file
    config_file = os.path.join(load_ckpt_dir, 'configs.pkl')
    config = helper.read_pickle(config_file)


    # Get the directory of the dataset
    base_dir = variables.fastmri_paths[config['data_args']['mri_type']]

    # Get the model type
    model_type = 'Classifier'

    config['data_args']['mask_box_augment'] = False

    # Get the data
    data = FastMRIAnnotated(base_dir,
                            batch_size=config['train_args']['batch_size'],
                            evaluating=True,
                            num_data_loader_workers=4,
                            **config['data_args'],
                            )
    data.prepare_data()
    data.setup()

    # Load the model
    model = helper.load_model(model_type, config, ckpt)


#%% Evaluate the model
    model.eval()
    model.cuda()

    uncalib_preds = []
    targets = []


    with torch.no_grad():
        for i in range(len(data.val)):
            x = data.val[i][0].to(model.device).unsqueeze(0)
            y = data.val[i][-1]

            # Get the isotonic and platt calibrated predictions
            ypred = model(x)
            yhat = torch.nn.functional.sigmoid(ypred)
            uncalib_preds.append(yhat.cpu())

            # Store the targets
            targets.append(y)

    uncalib_preds_gt = torch.clamp(torch.tensor(uncalib_preds).cpu().flatten() + 1e-8, 0, 1)
    targets_gt = torch.clamp(torch.tensor(targets).cpu(), 0, 1)


    # Get different classification metrics
    uncalib_classif = uncertainty_eval.get_classification_metrics(uncalib_preds_gt, targets_gt)

    # Combine all the metrics to be saved
    metrics = {'Uncalibrated': uncalib_classif}

    # Save the metrics
    with open(os.path.join(eval_dir, 'calibration_gt', 'metrics.yaml'), 'w') as f:
        documents = yaml.dump(metrics, f, sort_keys=False)




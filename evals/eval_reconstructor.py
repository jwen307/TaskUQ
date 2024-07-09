#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnf.py
    - Script to train a conditional normalizing flow
"""


import os
from pathlib import Path
import traceback

import sys
sys.path.append("../")

from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper
from evals import metrics, recon_metrics1
import variables
import yaml


#Get the input arguments
args = helper.flags()

#Get the checkpoint arguments if needed
load_ckpt_dir = args.load_ckpt_dir
load_last_ckpt = args.load_last_ckpt


if __name__ == "__main__":

    #Load the previous configurations
    ckpt_name = 'last.ckpt' if load_last_ckpt else 'best.ckpt'
    ckpt = os.path.join(load_ckpt_dir,
                        'checkpoints',
                        ckpt_name)

    #Get the configuration file
    config_file = os.path.join(load_ckpt_dir, 'configs.pkl')
    config = helper.read_pickle(config_file)
    

    try:
        # Get the directory of the dataset
        base_dir = variables.fastmri_paths[config['data_args']['mri_type']]

        # Get the model type
        model_type = load_ckpt_dir.split('/')[-3]

        # Load the model
        model = helper.load_model(model_type, config, ckpt)
        model.eval()

        # Use RSS for the knee dataset
        rss = True

        for accel in [2,4,8,16]:

            # Get the data
            data = FastMRIDataModule(base_dir,
                                     batch_size=config['train_args']['batch_size'],
                                     num_data_loader_workers=4,
                                     specific_accel = accel,
                                     **config['data_args'],
                                     )
            data.prepare_data()
            data.setup()

            print('Acceleration: ', accel)


            report_path = os.path.join(load_ckpt_dir, 'metrics_accel{0}'.format(accel))
            Path(report_path).mkdir(parents=True, exist_ok=True)


            # Get the metrics using the volume
            vol_metrics = recon_metrics1.evaluate(model, report_path, data, model_type=model_type)
            with open(os.path.join(report_path, 'metric_vol_accel.yaml'), 'w') as file:
                documents = yaml.dump(vol_metrics, file)

            # Get the FID
            if model_type == 'VarNet':
                get_cfid=False
            else:
                get_cfid=True
            fid_metrics = metrics.evaluate_posterior(model, data, num_samples=32, temp=1.0, rss=rss, test=False, get_cfid=get_cfid)
            with open(os.path.join(report_path, 'metrics_fid_accel.yaml'), 'w') as file:
                documents = yaml.dump(fid_metrics, file)



    except:

        traceback.print_exc()
       
        


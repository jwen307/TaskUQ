#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

train_cnf.py
    - Script to train a conditional normalizing flow
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers,seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os
from pathlib import Path
import traceback

import sys
sys.path.append("../")

from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper
import variables


#Get the input arguments
args = helper.flags()

#Get the checkpoint arguments if needed
load_ckpt_dir = args.load_ckpt_dir
load_last_ckpt = args.load_last_ckpt


if __name__ == "__main__":
    
    #Use new configurations if not loading a pretrained model
    if load_ckpt_dir == 'None':
        model_type = args.model_type

        #Get the configurations
        configuration = helper.select_config(model_type)
        config = configuration.config
        ckpt=None
    
    #Load the previous configurations
    else:
        ckpt_name = 'last.ckpt' if load_last_ckpt else 'best_bpd.ckpt'
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
        model_type = config['flow_args']['model_type']

        # Get the data
        data = FastMRIDataModule(base_dir,
                                 batch_size=config['train_args']['batch_size'],
                                 num_data_loader_workers=4,
                                 **config['data_args'],
                                 )
        data.prepare_data()
        data.setup()


        #Load the model
        model = helper.load_model(model_type, config, ckpt)

        # Compile the model (Doesn't work if there's complex numbers like in fft2c)
        #model = torch.compile(model)

        # Create the tensorboard logger
        Path(variables.log_dir).mkdir(parents=True, exist_ok=True)
        logger = loggers.TensorBoardLogger(variables.log_dir, name=model_type)

        # Create the checkpoint callback
        ckpt_callback = ModelCheckpoint(
            save_top_k = 1,
            monitor='val_bpd',
            mode = 'min',
            filename='best',
            )


        # Create the trainers (Note: MulticoilCNF does manual optimization)
        trainer = pl.Trainer(
            max_epochs=150,
            accelerator='gpu',
            logger=logger,
            check_val_every_n_epoch=1,
            callbacks=[ckpt_callback],
            strategy='ddp_find_unused_parameters_true',
            # limit_train_batches=16,
            # limit_val_batches=16,
        )


        # Save the configurations
        model_path = trainer.logger.log_dir
        Path(model_path).mkdir(parents=True, exist_ok=True)
        config_file = os.path.join(model_path, 'configs.pkl')
        helper.write_pickle(config, config_file)

        # Train the model
        if ckpt is None:
            print("Starting Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader())
            trainer.save_checkpoint(os.path.join(model_path,'checkpoints','last.ckpt'))

        else:
            print("Resuming Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader(),ckpt_path=ckpt)
            trainer.save_checkpoint(os.path.join(model_path,'checkpoints','last.ckpt'))


    except:

        traceback.print_exc()
       
        


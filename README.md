# Task-Driven Uncertainty Quantification in Inverse Problems via Conformal Prediction

## Description
This is the code for the paper [Task-Driven Uncertainty Quantification in Inverse Problems via Conformal Prediction](https://www.arxiv.org/abs/2405.18527).

## Installation
Please follow the instructions to setup the environment to run the repo.
1. Create a new environment with the following commands
```
conda create -n taskuq python=3.9 numpy=1.23 pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy cudatoolkit=11.8 cudnn cutensor nccl
conda install -c anaconda h5py=3.6.0
```
2. From the project root directory, install the requirements with the following command
```
pip install -r requirements.txt
```


## Usage Prerequisites
1. Download the fastMRI knee and brain datasets from [here](https://fastmri.org/)
2. Set the directory of the multicoil fastMRI knee and brain datasets to where they are stored on your device
    - Change [variables.py](variables.py) to set the paths to the dataset and your prefered logging folder
3. Change the configurations for training in the config files located in **train/configs/**. The current values are set to the ones used in the paper.


## Overview
- All models used can be found in the **models** folder
- The training scripts can be found in the **train** folder
- The evaluation scripts can be found in the **evals** folder
- The conformal prediction scripts can be found in the **conformal** folder
   - The underworkings of the conformal methods can be found in conformal_utils.py

## Training
First, set the directory to the **train** folder
```
cd train
```

To train a model, modify the configuration file in **train/configs/** and run the following commands for the model you want to train.
```
# Training a SIMCLR model
python pretrain_classifier_simclr.py

# Training a classifier model (note, specify a  pretrained SIMCLR model if desired in the config file)
python train_classifier.py

# Training a CNF model
python train_cnf.py --model_type MulticoilCNF 

# Training an E2E Model
python train_varnet.py
```

All models will be saved in the logging folder specified in [variables.py](variables.py)


## Evaluation
To get the evaluation metrics like PSNR, SSIM, and FID for reconstruction mdoels, run the following commands
```
# Navigate to the evals folder
cd evals

# Run the evaluation 
python eval_reconstructor.py --load_ckpt_dir <path to the model checkpoint>

#Example
python eval_cnf.py --load_ckpt_dir /home/user/mri_cnf/MulticoilCNF/version_0/
```

To perform the Monte Carlo evaluation, run the following commands
```
# Navigate to the conformal folder
cd conformal

# Run the evaluation
python eval_conformal_empirical.py --load_ckpt_dir <path to the classifier checkpoint> --load_recon_dir <path to the reconstructor checkpoint>
```
You can change the number of samples used, error rate, and other parameters in eval_conformal_empirical.py.

Finally, to run the multi-round sampling procedure, run the following commands
```
python multi_round_sampling.py --load_ckpt_dir <path to the classifier checkpoint> --load_recon_dir <path to the reconstructor checkpoint>
```



## Notes
- The first time using a dataset will invoke the preprocessing step required for compressing the coils. 
Subsequent runs will be much faster since this step can be skipped.


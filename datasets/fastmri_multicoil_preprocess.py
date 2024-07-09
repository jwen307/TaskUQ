#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fastmri_multicoil_preprocess.py
    - Preprocess the data to get the normalization value for each volume
"""


import torch
import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict
import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
import xml.etree.ElementTree as etree
import cupy as cp
from typing import Dict, Optional, Sequence, Tuple, Union
import yaml
from warnings import warn

import sys
sys.path.append("..")

from datasets.masks.mask import get_mask, apply_mask


# Coil compression and image cropping
# Needs to be [imgsize, imgsize, num_coils]
# Modified so it uses cupy and is 10x faster
def ImageCropandKspaceCompression(x, size, num_vcoils=8, vh=None):
    w_from = (x.shape[0] - size) // 2  # crop images into 384x384
    h_from = (x.shape[1] - size) // 2
    w_to = w_from + size
    h_to = h_from + size
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > num_vcoils:
        x_tocompression = cropped_x.reshape(size ** 2, cropped_x.shape[-1])

        if vh is None:
            # Convert to a cupy tensor
            with cp.cuda.Device(0):
                x_tocompression = cp.asarray(x_tocompression)
                U, S, Vh = cp.linalg.svd(x_tocompression, full_matrices=False)
                coil_compressed_x = cp.matmul(x_tocompression, Vh.conj().T)
                coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
            # x_tocompression = np.asarray(x_tocompression)
            # U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
            # coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
            # coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)

        else:
            coil_compressed_x = np.matmul(x_tocompression, vh.conj().T)
            coil_compressed_x = coil_compressed_x[:, 0:num_vcoils].reshape(size, size, num_vcoils)
            Vh = vh

    else:
        coil_compressed_x = cropped_x

    if vh is not None:
        return coil_compressed_x
    return cp.asnumpy(coil_compressed_x), cp.asnumpy(Vh)


# Function to compress and crop the data
def get_compressed(kspace, img_size, num_vcoils = 8, vh=None):
    '''
    kspace (tensor) [num_slices, num_vcoils, size0, size1] : the ground truth kspace
    img_size (int) : size to crop the image to
    num_vcoils (int) : number of virtual coils to compress to
    vh (tensor) : matrix with vh from (U,S,Vh) of SVD
    '''

    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace)
    
    compressed_imgs = []
    Vhs = []

    for i in range(kspace.shape[0]):
        #Compress to num_vcoils virtual coils and crop
        compressed_img, Vh = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img[i]).transpose(1,2,0),
                                                       img_size, num_vcoils, vh)
        compressed_imgs.append(to_tensor(compressed_img))
        Vhs.append(Vh)

    #Combine into one tensor stack
    compressed_imgs = torch.stack(compressed_imgs)
    Vhs = np.stack(Vhs)

    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(compressed_imgs.permute(0,3,1,2,4))
    
    if vh is None:
        return compressed_k, Vhs
    else:
        return compressed_k

# Function to get the normalizing value
def get_normalizing_val(kspace, img_size, mask_type, accel_rate):
    """
    kspace (tensor) [num_slices, num_coils, size0, size1]: ground truth kspace
    img_size (int): size of the image
    mask_type (str): type of mask
    accel_rate (int): acceleration rate of the mask
    """

    #Apply the mask
    mask = get_mask(accel=accel_rate, size=img_size, mask_type=mask_type)
    masked_kspace = apply_mask(kspace, mask)
    
    #Get the zf imgs
    masked_imgs = fastmri.ifft2c(masked_kspace)
    
    #Get the magnitude imgs for the zf imgs
    zf_mags = fastmri.complex_abs(masked_imgs)
    # Note: Should be this but was trained with the above
    #zf_mags = fastmri.rss_complex(masked_imgs, dim=1)

    #Normalized based on the 95th percentile max value of the magnitude
    max_val = np.percentile(zf_mags.cpu(), 95)

    return masked_kspace, max_val

# Function to preprocess data and find the normalizing value
def preprocess_data(base_dir, **kwargs):
    '''
    base_dir (str): Directory of the dataset (Ex: "/storage/fastMRI_brain/data/")
    kwargs: Dataset parameters from the configs files
    '''

    #Parameters
    #mri_type = kwargs['mri_type']
    accel_rate = kwargs['accel_rate']
    img_size = kwargs['img_size']
    num_vcoils = kwargs['num_vcoils']
    challenge = 'multicoil' #kwargs['challenge']
    mask_type = kwargs['mask_type']
    acquisition = kwargs['scan_type']
    mri_type = kwargs['mri_type']

    dataset_types = ['train', 'val']
    if mri_type=='brain':
        dataset_types.append('small_T2_test')
    for dataset_type in dataset_types:
        if not dataset_type == 'small_T2_test':
            dataset_dir = os.path.join(base_dir, '{0}_{1}'.format(challenge, dataset_type))
        else:
            dataset_dir = os.path.join(base_dir, dataset_type)

        # If the mask type is progressive, this means acceleration rate is a list
        #if mask_type =='prog':
        if isinstance(accel_rate, list):
            accel_rate = kwargs['accel_rate'][-1]

        new_dir = os.path.join(base_dir, '{0}_{1}_{2}coils_{3}accel_preprocess_{4}'.format(challenge, dataset_type, num_vcoils,accel_rate, mask_type))

        #Create the directory if it does not already exist and preprocess the files
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

            # Get the files
            files = list(Path(dataset_dir).iterdir())

            for fname in tqdm(sorted(files)):

                #Skip non-data files
                if fname.name[0] == '.':
                    continue

                #Recover the metadata
                metadata, num_slices = retrieve_metadata(fname)

                with h5py.File(fname, "r") as hf:

                    #Get the kspace
                    kspace = hf['kspace']

                    #Get the attributes of the volume
                    attrs = dict(hf.attrs)
                    attrs.update(metadata)

                    #if attrs['acquisition'] != acquisition or attrs['encoding_size'][1] < img_size:
                    #    continue

                    if attrs['encoding_size'][1] < img_size:
                        continue

                    kspace_torch = to_tensor(kspace[:])

                    # Make sure each image has at least the number of virtual coils
                    if kspace_torch.shape[1] <= num_vcoils:
                        continue


                    #Get the virtual coils and the normalization value for the volume
                    compressed_k, vhs = get_compressed(kspace_torch, img_size, num_vcoils)
                    masked_kspace, max_val = get_normalizing_val(compressed_k, img_size, mask_type, accel_rate)


                if num_slices != vhs.shape[0]:
                    raise Exception('Problem with {}'.fname.name[0])


                #Save the processed data into the new h5py file
                with h5py.File(os.path.join(new_dir, fname.name), 'w') as nf:
                    nf.attrs['max_val'] = max_val
                    #Save the Vh from the svd
                    vh = nf.create_dataset('vh', vhs.shape, data=vhs)


        
#%% Helper functions from the fastMRI repository
#https://github.com/facebookresearch/fastMRI
def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(
        key: str, data_config_file: Union[str, Path, os.PathLike] = "fastmri_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.
    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.
    Args:
        key: key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file: Optional; Default path configs file to fetch path
            from.
    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "/path/to/knee",
            "brain_path": "/path/to/brain",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path configs at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


def retrieve_metadata(fname):
    with h5py.File(fname, "r") as hf:
        et_root = etree.fromstring(hf["ismrmrd_header"][()])

        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )
        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (
            int(et_query(et_root, rec + ["x"])),
            int(et_query(et_root, rec + ["y"])),
            int(et_query(et_root, rec + ["z"])),
        )

        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max

        num_slices = hf["kspace"].shape[0]

    metadata = {
        "padding_left": padding_left,
        "padding_right": padding_right,
        "encoding_size": enc_size,
        "recon_size": recon_size,
    }

    return metadata, num_slices
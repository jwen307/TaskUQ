#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import requests
import torch
import pytorch_lightning as pl
import os
from pathlib import Path
import h5py
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from typing import Dict, Optional, Sequence, Tuple, Union

import sigpy as sp
import sigpy.mri as mr

import sys
sys.path.append("..")

import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np
from util import viz, network_utils


class FastMRIAnnotated(pl.LightningDataModule):

    def __init__(self, base_path, batch_size: int = 32, num_data_loader_workers: int = 4, evaluating = False, bounding_boxes = False, **kwargs):
        """
        Initialize the data module for the fastMRI dataset.

        Parameters
        ----------
        base_path : str
            Location of the dataset (Ex: "/storage/fastMRI_brain/data/")
        batch_size : int, optional
            Size of a mini batch.
            The default is 4.
        num_data_loader_workers : int, optional
            Number of workers.
            The default is 8.

        Returns
        -------
        None.
        """
        super().__init__()

        self.base_path = base_path
        self.batch_size = batch_size
        self.num_data_loader_workers = num_data_loader_workers

        self.img_size = kwargs['img_size']
        self.complex = kwargs['complex']
        self.challenge = kwargs['challenge']
        self.mri_type = kwargs['mri_type'] #'knee', 'brain'
        self.scan_type = kwargs['scan_type']
        self.slice_range = kwargs['slice_range']
        self.specific_label = kwargs['specific_label']
        self.augmented = kwargs['augmented']

        if 'contrastive' in kwargs:
            self.contrastive = kwargs['contrastive']
            if self.contrastive:
                self.augmented = False
        else:
            self.contrastive = False

        if 'mask_box_augment' in kwargs:
            self.mask_box_augment = kwargs['mask_box_augment']
        else:
            self.mask_box_augment = False


        self.num_vcoils = kwargs['num_vcoils'] if 'num_vcoils' in kwargs else None
        self.accel_rate = kwargs['accel_rate'] if 'accel_rate' in kwargs else None

        self.evaluating = evaluating
        self.bounding_boxes = bounding_boxes



    def prepare_data(self):
        """
        Preparation steps like downloading etc.
        Don't use self here!

        Returns
        -------
        None.

        """
        None

    def setup(self, stage: str = None):
        """
        This is called by every GPU. Self can be used in this context!

        Parameters
        ----------
        stage : str, optional
            Current stage, e.g. 'fit' or 'test'.
            The default is None.

        Returns
        -------
        None.

        """

        # Get the directory of the training and validation set
        if self.complex:
            train_dir = os.path.join(self.base_path,'{0}_{1}_{2}coils_{3}accel'.format('singlecoil',
                                                                  'train',
                                                                  self.num_vcoils,
                                                                  self.accel_rate))
            val_dir = os.path.join(self.base_path,'{0}_{1}_{2}coils_{3}accel'.format('singlecoil',
                                                                    'val',
                                                                    self.num_vcoils,
                                                                    self.accel_rate))
        else:
            train_dir = os.path.join(self.base_path, '{0}_{1}'.format(self.challenge, 'train'))
            val_dir = os.path.join(self.base_path, '{0}_{1}'.format(self.challenge, 'val'))


        self.train = SinglecoilDatasetAnnotated(self.base_path,
                                                    train_dir,
                                                    self.mri_type,
                                                    self.img_size,
                                                    self.scan_type,
                                                    self.slice_range,
                                                    self.specific_label,
                                                    augment = self.augmented,
                                                    contrastive=self.contrastive,
                                                    give_bb=self.bounding_boxes,
                                                    complex=self.complex,
                                                    )


        # Get the sample weights for the weighted random sampler
        if not self.contrastive and not self.mask_box_augment:
            if not self.evaluating:
                print('Calculating sample weights')
                label_counts= []
                for i in range(len(self.train)):
                    data = self.train[i]
                    label_counts.append(data[-1])
                self.num_pos = sum(label_counts)
                self.num_neg = len(self.train) - self.num_pos
                class_counts = [self.num_neg, self.num_pos]
                self.sample_weights = [1/class_counts[i] for i in label_counts]

        if self.mri_type=='knee':
            self.slice_range = 0.8

        # Get the validation dataset
        self.val = SinglecoilDatasetAnnotated(self.base_path,
                                              val_dir,
                                              self.mri_type,
                                              self.img_size,
                                               self.scan_type,
                                              self.slice_range,
                                              self.specific_label,
                                              augment=False,
                                              contrastive=self.contrastive,
                                              give_bb=self.bounding_boxes,
                                              complex=self.complex,
                                              )






    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """

        # Create the weighted random sampler
        if self.contrastive or self.evaluating or self.mask_box_augment:
            return DataLoader(self.train, batch_size=self.batch_size,
                                num_workers=self.num_data_loader_workers,
                                shuffle=True, pin_memory=False)

        #elif self.augmented:
        sampler = torch.utils.data.WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self.train), replacement=True)

        return DataLoader(self.train, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=False, sampler=sampler)




    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.val, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=False)






class SinglecoilDatasetAnnotated(torch.utils.data.Dataset):
    def __init__(self, base_dir, root, mri_type, img_size=320, scan_type=None,
                 slice_range=None, specific_label=None, augment=False, contrastive=False, complex=False,
                 give_bb = False, **kwargs):
        '''
        scan_type: None, 'CORPD_FBK', 'CORPDFS_FBK' for knee
        scan_type: None, 'AXT2'
        '''

        self.root = root
        self.img_size = img_size
        self.examples = []
        self.complex = complex

        self.slice_range = slice_range
        self.augment = augment
        self.contrastive = contrastive
        self.bounding_boxes = give_bb


        #Pull the annotations
        annotations_csv, annotations_list_csv = get_annotations(base_dir, mri_type)

        if self.augment:
            transforms = [
                torchvision.transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                #torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(size=320, scale=(0.8, 1.0), antialias=True),
                #torchvision.transforms.GaussianBlur(kernel_size=7),
                ]

            if not self.complex:
                transforms.append(torchvision.transforms.RandomApply([
                    v2.ColorJitter(brightness=0.5,
                                                       #contrast=0.5,
                                                       #saturation=0.5,
                                                       )
                ], p=0.5),)

            self.transform = torchvision.transforms.Compose(transforms)

        elif self.contrastive:
            contrast_transforms = [torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                       torchvision.transforms.RandomResizedCrop(size=320, scale=(0.8, 1.0), antialias=True),
                                                        #torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                        torchvision.transforms.RandomAffine(degrees=(-15, 15),
                                                                                              translate=(
                                                                                              0.1, 0.1)),
                                                       torchvision.transforms.RandomApply([
                                                           v2.ColorJitter(brightness=0.5,
                                                                                  #contrast=0.5,
                                                                                  #saturation=0.5,
                                                                                  )
                                                       ], p=0.8),
                                                       # transforms.RandomGrayscale(p=0.2),
                                                       torchvision.transforms.GaussianBlur(kernel_size=7),
                                                       #torchvision.transforms.ToTensor(),
                                                       # transforms.Normalize((0.5,), (0.5,))
                                                       ]

            if not self.complex:
                contrast_transforms.append(torchvision.transforms.RandomApply([
                                                           torchvision.transforms.ColorJitter(brightness=0.5,
                                                                                  contrast=0.5,
                                                                                  #saturation=0.5,
                                                                                  )
                                                       ], p=0.8))

            contrast_transforms = torchvision.transforms.Compose(contrast_transforms)

            self.transform = ContrastiveTransformations(contrast_transforms, n_views=2)


        files = list(Path(root).iterdir())
        self.total_labels = 0

        print('Loading Data')
        for fname in tqdm(sorted(files)):

            # Skip non-data files
            if fname.name[0] == '.':
                continue

            #Check if the file was annotated
            if fname.stem not in annotations_list_csv.values:
                continue

            # Recover the metadata
            #metadata, num_slices = retrieve_metadata(fname)

            with h5py.File(fname, "r") as hf:

                # Get the attributes of the volume
                attrs = dict(hf.attrs)
                #attrs.update(metadata)

                if scan_type is not None:
                    if attrs["acquisition"] != scan_type:
                        continue


                recon = hf['reconstruction_rss'] if not self.complex else hf['gt']

                # Get the number of slices
                #num_slices = hf['reconstruction_rss'].shape[0]
                num_slices = recon.shape[0]

                # Check if the shape of the reconstruction is correct
                #if hf['reconstruction_rss'].shape[-1] != self.img_size:
                if recon.shape[-1] != self.img_size:
                    continue

                # Use all the slices if a range is not specified
                if self.slice_range is None:
                    slice_range = [0, num_slices]
                else:
                    if type(self.slice_range) is list:
                        if self.slice_range[1] - self.slice_range[0] > num_slices:
                            slice_range = [0,num_slices]
                        else:
                            slice_range = self.slice_range
                    elif self.slice_range < 1.0:

                        # Use percentage of center slices (i.e. center 80% of slices)
                        slice_range = [int(num_slices * (1 - self.slice_range)), int(num_slices * self.slice_range)]

                labels = []
                bounding_boxes = []
                all_bounding_boxes = []
                all_labels = []
                for k in range(num_slices):
                    # Get the specific lines in the annotations for the file
                    annotation_df = annotations_csv[
                        (annotations_csv["file"] == fname.stem)
                        & (annotations_csv["slice"] == k)
                        ]

                    # If you've specified a specific label to do classification
                    if specific_label is not None:
                        # Cobmining laels
                        if isinstance(specific_label,list):

                            if any([label in annotation_df['label'].values for label in specific_label]):
                                labels.append(1)
                                # _, _, _, x0, y0, w, h, label_txt = annotation_df[annotation_df['label']==specific_label].values.tolist()[0]
                                # bounding_boxes.append([x0,y0,w,h])
                                slice_bb = []
                                for l in range(len(annotation_df[annotation_df['label'].isin(specific_label)].values.tolist())):
                                    _, _, _, x0, y0, w, h, label_txt = annotation_df[annotation_df['label'].isin(specific_label)].values.tolist()[l]
                                    slice_bb.append([x0,y0,w,h])
                                bounding_boxes.append(slice_bb)

                            else:
                                labels.append(0)
                                bounding_boxes.append([[0,0,0,0]])
                        else:
                            if specific_label in annotation_df['label'].values:
                                labels.append(1)
                                # _, _, _, x0, y0, w, h, label_txt = annotation_df[annotation_df['label']==specific_label].values.tolist()[0]
                                # bounding_boxes.append([x0,y0,w,h])
                                slice_bb = []
                                for l in range(len(annotation_df[annotation_df['label']==specific_label].values.tolist())):
                                    _, _, _, x0, y0, w, h, label_txt = annotation_df[annotation_df['label']==specific_label].values.tolist()[l]
                                    slice_bb.append([x0,y0,w,h])
                                bounding_boxes.append(slice_bb)

                            else:
                                labels.append(0)
                                bounding_boxes.append([[0,0,0,0]])

                    # Get all of the pathologies
                    slice_all_bb = []
                    bb_labels = []
                    for v in range(len(annotation_df['label'].values.tolist())):
                        _, _, _, x0, y0, w, h, label_txt = annotation_df.values.tolist()[v]
                        slice_all_bb.append([x0,y0,w,h])
                        bb_labels.append(label_txt)
                    all_bounding_boxes.append(slice_all_bb)
                    all_labels.append(bb_labels)


                self.total_labels += np.sum(labels)

            #print(f'Length of labels: {len(labels)}')
            #print(f'Length of bounding boxes: {len(bounding_boxes)}')
            #print(f'Length of slice range: {slice_range[1]-slice_range[0]}')
            self.examples += [(fname, slice_ind, labels[slice_ind], bounding_boxes[slice_ind], all_bounding_boxes[slice_ind], all_labels[slice_ind])
                              for slice_ind in range(slice_range[0], slice_range[1])]
        print('Total Number of Positive Labels: {}'.format(self.total_labels))
        print('Total Number of Labels: {}'.format(len(self.examples)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice, label, bounding_box, all_bounding_boxes, all_labels = self.examples[i]

        with h5py.File(fname, "r") as hf:
            # Get the compressed target kspace
            recon = hf['reconstruction_rss'][dataslice] if not self.complex else hf['gt'][dataslice]
            recon = to_tensor(recon).unsqueeze(0) if not self.complex else to_tensor(recon)

            # Augment if selected
            if self.augment:
                recon = self.transform(recon).float()

            # Contrastive if selected
            elif self.contrastive:
                recon = self.transform(recon)

            acquisition = hf.attrs['acquisition']

            bb = {'bounding_box': torch.as_tensor(bounding_box, dtype=torch.float32),
                    'all_bounding_boxes': torch.as_tensor(all_bounding_boxes, dtype=torch.float32),
                    'all_labels': all_labels}


        if self.bounding_boxes:
            return (
                recon,
                acquisition,
                fname.name,
                dataslice,
                bb,
                label
            )
        else:
            return (
                recon,
                acquisition,
                fname.name,
                dataslice,
                label
            )




# From FastMRI
def download_csv(version, subsplit, path, get_file_list=False):
    # request file by git hash and mri type
    if not get_file_list:
        if version is None:
            url = f"https://raw.githubusercontent.com/microsoft/fastmri-plus/main/Annotations/{subsplit}.csv"
        else:
            url = f"https://raw.githubusercontent.com/microsoft/fastmri-plus/{version}/Annotations/{subsplit}.csv"
    else:
        if version is None:
            url = f"https://raw.githubusercontent.com/microsoft/fastmri-plus/main/Annotations/{subsplit}_file_list.csv"
        else:
            url = f"https://raw.githubusercontent.com/microsoft/fastmri-plus/{version}/Annotations/{subsplit}_file_list.csv"

    request = requests.get(url, timeout=10, stream=True)

    # create temporary folders
    #Path(path).mkdir(parents=True, exist_ok=True)

    # download csv from github and save it locally
    with open(path, "wb") as fh:
        for chunk in request.iter_content(1024 * 1024):
            fh.write(chunk)
    return path

def get_annotations(base_dir, mri_type, annotation_version=None):
    # download csv file from github using git hash to find certain version
    annotation_name = f"{mri_type}.csv"
    Path(base_dir, "annotations").mkdir(parents=True, exist_ok=True)
    annotation_path = Path(base_dir, "annotations", annotation_name)
    if not annotation_path.is_file():
        annotation_path = download_csv(
            annotation_version, mri_type, annotation_path
        )

    # Download the csv file of volumes annotated
    annotation_name = f"{mri_type}_file_list.csv"
    annotation_list_path = Path(base_dir, "annotations", annotation_name)
    if not annotation_list_path.is_file():
        annotation_list_path = download_csv(
            annotation_version, mri_type, annotation_list_path,
            get_file_list=True
        )

    annotations_csv = pd.read_csv(annotation_path)
    annotations_list_csv = pd.read_csv(annotation_list_path, header=None)

    return annotations_csv, annotations_list_csv


# Transformations for contrastive learning. From https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


# Example usage
if __name__ == '__main__':

    kwargs = {
        'mri_type': 'knee',  # brain or knee
        'center_frac': 0.08,
        'accel_rate': 8,
        'img_size': 320,
        'challenge': "singlecoil",
        'complex': False,  # if singlecoil, specify magnitude or complex
        'scan_type': None,  # Knee: 'CORPD_FBK' Brain: 'AXT2'
        'mask_type': 'knee',  # Options :'s4', 'default', 'center_aug'
        'num_vcoils': 8,
        'acs_size': 9,  # 13 for knee, 32 for brain
        'slice_range': None,  # [0, 8], None
        'augmented': True,
        'specific_label': 'Meniscus Tear',
        'mask_box_augment': True,
    }

    # Location of the dataset
    if kwargs['mri_type'] == 'brain':
        base_dir = "/storage/fastMRI_brain/data/"
    elif kwargs['mri_type'] == 'knee':
        base_dir = "/storage/fastMRI/data/"
    else:
        raise Exception("Please specify an mri_type in configs")

    data = FastMRIAnnotated(base_dir, batch=4, evaluating=True, **kwargs)
    data.prepare_data()
    data.setup()
    # dataset = MulticoilDataset(dataset_dir, max_val_dir, img_size, mask_type)

    dataset = data.val
    i = -14
    img = dataset[i][0]
    y = dataset[i][-1]

    if kwargs['complex']:
        viz.show_img(network_utils.get_magnitude(img.unsqueeze(0)))
    else:
        viz.show_img(img.unsqueeze(0))



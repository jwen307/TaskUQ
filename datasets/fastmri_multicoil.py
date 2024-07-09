#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fastmri_multicoil.py
    - Contains the PL data module for multicoil fastMRI data
    - Contains helper functions for the data
"""


import torch
import pytorch_lightning as pl
import os
from pathlib import Path
import h5py
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np




import sys
sys.path.append("..")

from util import viz, network_utils
from datasets.masks.mask import get_mask, apply_mask
from datasets.fastmri_multicoil_preprocess import preprocess_data, ImageCropandKspaceCompression, retrieve_metadata
from datasets.fastmri_annotated import get_annotations
from datasets.fastmri_annotated import ContrastiveTransformations




# PyTorch Lightning Data Module for multicoil fastMRI data
class FastMRIDataModule(pl.LightningDataModule):

    def __init__(self, base_path, batch_size: int = 32, num_data_loader_workers: int = 4, evaluating=False, specific_accel= None, **kwargs):
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

        self.accel_rate = kwargs['accel_rate']
        self.img_size = kwargs['img_size']
        self.use_complex = kwargs['complex']
        #kwargs['challenge'] = 'multicoil' # Only use the multicoil data. Combine for the single coil case
        self.challenge = kwargs['challenge']
        self.acs = kwargs['acs_size']
        self.evaluating = evaluating


        #Number of virtual coils
        self.num_vcoils = kwargs['num_vcoils']


        # Define a subset of scan types to use
        if 'scan_type' in kwargs:
            self.scan_type = kwargs['scan_type']
        else:
            self.scan_type = None

        # Define a subset of slices to use
        if 'slice_range' in kwargs:
            self.slice_range = kwargs['slice_range']
        else:
            self.slice_range = None

        # Define the type of mri image
        if 'mri_type' in kwargs:
            self.mri_type = kwargs['mri_type']
        else:
            self.mri_type = 'knee'

        # Define the pathology to look for
        if 'specific_label' in kwargs:
            self.specific_label = kwargs['specific_label']
        else:
            self.specific_label = None

        if 'augmented' in kwargs:
            self.augment = kwargs['augmented']
        else:
            self.augment = False

        if 'contrastive' in kwargs:
            self.contrastive = kwargs['contrastive']
        else:
            self.contrastive = False

        if 'mask_type' in kwargs:
            self.mask_type = kwargs['mask_type']
        else:
            self.mask_type = self.mri_type


        self.specific_accel = specific_accel

        # Preprocess the data if not done so already
        preprocess_data(self.base_path, **kwargs)

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



        train_dir = os.path.join(self.base_path, '{0}_{1}'.format('multicoil', 'train'))
        val_dir = os.path.join(self.base_path, '{0}_{1}'.format('multicoil', 'val'))

        if self.mask_type == 'prog':
            max_val_accel = self.accel_rate[-1]
        else:
            max_val_accel = self.accel_rate

        max_val_dir_train = os.path.join(self.base_path,
                                         '{0}_{1}_{2}coils_{3}accel_preprocess_{4}'.format('multicoil', 'train', self.num_vcoils, max_val_accel, self.mask_type))
        max_val_dir_val = os.path.join(self.base_path,
                                       '{0}_{1}_{2}coils_{3}accel_preprocess_{4}'.format('multicoil', 'val', self.num_vcoils, max_val_accel, self.mask_type))

        # For evaluation, specific a specific acceleration rate
        if self.specific_accel is not None:
            self.accel_rate = self.specific_accel


        # Assign train/val datasets for use in dataloaders
        self.train = MulticoilDataset(train_dir,
                                      max_val_dir_train,
                                      self.img_size, self.mask_type,
                                      self.accel_rate, self.scan_type,
                                      self.num_vcoils,
                                      self.slice_range,
                                      self.specific_label,
                                      self.acs,
                                      self.augment,
                                      self.contrastive,
                                      self.challenge,
                                      self.mri_type
                                      )

        # Limit the slice range for the evaluation to remove edge slices with poor SNR
        if self.mri_type == 'brain':
            self.slice_range = [0,8]
        elif self.mri_type == 'knee':
            self.slice_range = 0.8

        if self.specific_label is not None:
            if not self.contrastive:
                if not self.evaluating:
                    sample_weight_dir = os.path.join(self.base_path, 'sample_weights.npy')
                    print('Calculating sample weights')
                    if not os.path.exists(sample_weight_dir):
                        label_counts= []
                        for i in tqdm(range(len(self.train))):
                            data = self.train[i]
                            label_counts.append(data[-1])
                        self.num_pos = sum(label_counts)
                        self.num_neg = len(self.train) - self.num_pos
                        class_counts = [self.num_neg, self.num_pos]
                        self.sample_weights = [1/class_counts[i] for i in label_counts]

                        # Save the sample weights to be used next time
                        np.save(sample_weight_dir, self.sample_weights)
                    else:
                        self.sample_weights = np.load(sample_weight_dir)


        self.val = MulticoilDataset(val_dir,
                                    max_val_dir_val,
                                    self.img_size, self.mask_type,
                                    self.accel_rate, self.scan_type,
                                    self.num_vcoils,
                                    self.slice_range,
                                    self.specific_label,
                                    self.acs,
                                    augment=False,
                                    contrastive=self.contrastive,
                                    challenge=self.challenge,
                                    mri_type = self.mri_type
                                    )



        if self.mri_type == 'brain':
            self.slice_range = [0,6]
            test_dir = os.path.join(self.base_path, 'small_T2_test')

            max_val_dir_test = os.path.join(self.base_path,
                                           '{0}_{1}_{2}coils_{3}accel_preprocess_{4}'.format('multicoil', 'small_T2_test',
                                                                                         self.num_vcoils,
                                                                                         max_val_accel,
                                                                                             self.mask_type))

            #max_val_dir_test = os.path.join(self.base_path,
            #                                 'multicoil_small_T2_test_8coils')

            self.test = MulticoilDataset(test_dir,
                                    max_val_dir_test,
                                    self.img_size, self.mask_type,
                                    self.accel_rate, self.scan_type,
                                    self.num_vcoils,
                                    self.slice_range,
                                    self.specific_label,
                                    self.acs,
                                    augment=False,
                                    contrastive=self.contrastive,
                                    challenge=self.challenge,
                                    mri_type = self.mri_type
                                    )

        else:
            # Get a random subset of slices for testing but save them for later
            test_idx_file = os.path.join(self.base_path, 'test_idxs.npy')
            if os.path.exists(test_idx_file):
                self.test_idxs = np.load(test_idx_file)
            else:
                self.test_idxs = torch.randperm(len(self.val))[0:72].numpy()
                np.save(test_idx_file, self.test_idxs)

            self.test = torch.utils.data.Subset(self.val, self.test_idxs)


    def train_dataloader(self, percent_trainset=1.0):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        # Create the weighted random sampler
        if self.contrastive or self.evaluating or self.specific_label is None:
            if percent_trainset < 1.0:
                print('Using a subset of the training set: {0}%'.format(percent_trainset * 100))
                return DataLoader(torch.utils.data.Subset(self.train, torch.randperm(len(self.train))[:int(len(self.train) * percent_trainset)]),
                                  batch_size=self.batch_size,
                                  num_workers=self.num_data_loader_workers,
                                  shuffle=True, pin_memory=False)

            return DataLoader(self.train, batch_size=self.batch_size,
                              num_workers=self.num_data_loader_workers,
                              shuffle=True, pin_memory=False)

        # elif self.augmented:
        print('Using weighted random sampler')
        sampler = torch.utils.data.WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self.train),
                                                         replacement=True)

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
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)


# Dataset class for multicoil MRI data
class MulticoilDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_val_dir, img_size=320, mask_type='s4', accel_rate=4, scan_type=None, num_vcoils=8,
                 slice_range=None, specific_label=False, acs=13, augment=False, contrastive=False, challenge='multicoil', mri_type='knee', **kwargs):
        '''
        scan_type: None, 'CORPD_FBK', 'CORPDFS_FBK' for knee
        scan_type: None, 'AXT2' for brain
        '''

        self.root = root
        self.img_size = img_size
        self.mask_type = mask_type
        self.accel_rate = accel_rate
        self.max_val_dir = max_val_dir
        self.specific_label = specific_label
        self.acs = acs
        self.augment = augment
        self.contrastive = contrastive
        self.challenge = challenge
        self.examples = []
        self.mri_type = mri_type

        self.num_volumes = 0

        if self.specific_label is not None:
            base_dir = os.path.dirname(self.root)
            # Pull the annotations
            annotations_csv, annotations_list_csv = get_annotations(base_dir, mri_type)

        self.multicoil_transf = MulticoilTransform(mask_type=self.mask_type,
                                                   img_size=self.img_size,
                                                   accel_rate=self.accel_rate,
                                                   num_vcoils=num_vcoils,
                                                   )

        self.slice_range = slice_range

        if self.augment:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomResizedCrop(size=320, scale=(0.8, 1.0), antialias=True),
                torchvision.transforms.GaussianBlur(kernel_size=9),

                ])
        elif self.contrastive:
            contrast_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                                  torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                       torchvision.transforms.RandomResizedCrop(size=320, scale=(0.8, 1.0), antialias=True),
                                                      torchvision.transforms.RandomAffine(degrees=(-15, 15),
                                                                                          translate=(
                                                                                          0.1, 0.1)),
                                                       # transforms.RandomGrayscale(p=0.2),
                                                       torchvision.transforms.GaussianBlur(kernel_size=9),
                                                       #torchvision.transforms.ToTensor(),
                                                       # transforms.Normalize((0.5,), (0.5,))
                                                       ])

            self.transform = ContrastiveTransformations(contrast_transforms, n_views=2)

        files = list(Path(root).iterdir())
        total_labels = 0

        print('Loading Data')
        for fname in tqdm(sorted(files)):

            # Skip non-data files
            if fname.name[0] == '.':
                continue

            if self.specific_label is not None:
                # Check if the file was annotated
                if fname.stem not in annotations_list_csv.values:
                    continue

            # Recover the metadata
            metadata, num_slices = retrieve_metadata(fname)

            with h5py.File(fname, "r") as hf:

                # Get the attributes of the volume
                attrs = dict(hf.attrs)
                attrs.update(metadata)

                # Get volumes of the specific scan type
                if scan_type is not None:
                    if isinstance(scan_type,list):
                        if attrs["acquisition"] not in scan_type:
                            continue
                    else:
                        if attrs["acquisition"] != scan_type:
                            continue



                if attrs['encoding_size'][1] < img_size or hf['kspace'].shape[1] <= num_vcoils:
                    continue

                # Use all the slices if a range is not specified
                if self.slice_range is None:
                    num_slices = hf['kspace'].shape[0]
                    slice_range = [0, num_slices]
                else:
                    if type(self.slice_range) is list:
                        slice_range = self.slice_range

                        # Check to make sure the slice range is valid for this file
                        if slice_range[1] > hf['kspace'].shape[0]:
                            slice_range[1] = hf['kspace'].shape[0]

                    elif self.slice_range < 1.0:
                        num_slices = hf['kspace'].shape[0]
                        # Use percentage of center slices (i.e. center 80% of slices)
                        slice_range = [int(num_slices * (1 - self.slice_range)), int(num_slices * self.slice_range)]

                # Skip certain brain volumes to conform with evaluation
                skip_volumes = ['file_brain_AXT2_209_2090296', 'file_brain_AXT2_200_2000250', 'file_brain_AXT2_201_2010106',
                                'file_brain_AXT2_204_2130024','file_brain_AXT2_210_2100025']


                if str(fname.stem) in skip_volumes:
                    continue

                self.num_volumes += 1

                # Get the labels
                if self.specific_label is not None:
                    labels = []
                    bounding_boxes = []
                    for k in range(num_slices):
                        # Get the specific lines in the annotations for the file
                        annotation_df = annotations_csv[
                            (annotations_csv["file"] == fname.stem)
                            & (annotations_csv["slice"] == k)
                            ]

                        # If you've specified a specific label to do classification
                        if self.specific_label is not None:

                            # Cobmining laels
                            if isinstance(self.specific_label, list):

                                if any([label in annotation_df['label'].values for label in specific_label]):
                                    labels.append(1)
                                    # _, _, _, x0, y0, w, h, label_txt = annotation_df[annotation_df['label']==specific_label].values.tolist()[0]
                                    # bounding_boxes.append([x0,y0,w,h])
                                    slice_bb = []
                                    for l in range(len(annotation_df[annotation_df['label'].isin(
                                            specific_label)].values.tolist())):
                                        _, _, _, x0, y0, w, h, label_txt = \
                                        annotation_df[annotation_df['label'].isin(specific_label)].values.tolist()[l]
                                        slice_bb.append([x0, y0, w, h])
                                    bounding_boxes.append(slice_bb)

                                else:
                                    labels.append(0)
                                    bounding_boxes.append([[0, 0, 0, 0]])

                            else:
                                if self.specific_label in annotation_df['label'].values:
                                    labels.append(1)
                                    _, _, _, x0, y0, w, h, label_txt = annotation_df[annotation_df['label'] == specific_label].values.tolist()[0]
                                    bounding_boxes.append([[x0, y0, w, h]])
                                    # slice_bb = []
                                    # for l in range(
                                    #         len(annotation_df[annotation_df['label'] == specific_label].values.tolist())):
                                    #     _, _, _, x0, y0, w, h, label_txt = \
                                    #     annotation_df[annotation_df['label'] == specific_label].values.tolist()[l]
                                    #     slice_bb.append([x0, y0, w, h])
                                    # bounding_boxes.append(slice_bb)
                                else:
                                    labels.append(0)
                                    bounding_boxes.append([[0, 0, 0, 0]])

                    total_labels += np.sum(labels)
                    self.examples += [(fname, slice_ind, labels[slice_ind], bounding_boxes[slice_ind]) for slice_ind in
                                      range(slice_range[0], slice_range[1])]
                else:
                    self.examples += [(fname, slice_ind) for slice_ind in range(slice_range[0], slice_range[1])]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        if self.specific_label is not None:
            fname, dataslice, label, bounding_box = self.examples[i]
        else:
            fname, dataslice = self.examples[i]

        # Get the normalizing value and vh
        with h5py.File(os.path.join(self.max_val_dir, fname.name), 'r') as hf:
            max_val = hf.attrs['max_val']
            vh = hf['vh'][dataslice]

        with h5py.File(fname, "r") as hf:
            # Get the compressed target kspace
            kspace = hf['kspace'][dataslice]

            acquisition = hf.attrs['acquisition']

            zf_img, gt_img, mask = self.multicoil_transf(kspace=kspace,
                                                         max_val=max_val,
                                                         vh=vh)

            max_val = torch.tensor(max_val).float()

            # Coil combine if needed
            if self.challenge == 'singlecoil':

                maps = network_utils.get_maps(zf_img, self.acs, max_val)
                gt_img = network_utils.unnormalize(gt_img, max_val)
                gt_img = network_utils.multicoil2single(gt_img, maps).permute(0, 1, 4, 2, 3).squeeze(0)
                #Renormalize after combining
                gt_img = network_utils.normalize(gt_img, max_val)

            zf_img = zf_img.squeeze(0).float()
            gt_img = gt_img.squeeze(0).float()

            # Augment if selected
            if self.augment:
                gt_img = self.transform(gt_img).float()

            # Contrastive if selected
            elif self.contrastive:
                gt_img = self.transform(gt_img)



        if self.specific_label is not None:
            return (
                zf_img,
                gt_img,
                mask,
                # np.float32(max_val),
                max_val,
                acquisition,
                fname.name,
                dataslice,
                bounding_box,
                label,
            )
        else:
            return (
                zf_img.float(),
                gt_img.float(),
                mask,
                #np.float32(max_val),
                max_val,
                acquisition,
                fname.name,
                dataslice,
            )

# Transform for the multicoil dataset
class MulticoilTransform:

    def __init__(self, mask_type=None, img_size=320, accel_rate=4, num_vcoils=8):
        self.mask_type = mask_type
        self.img_size = img_size
        self.accel_rate = accel_rate
        self.num_vcoils = num_vcoils

    def __call__(self, kspace, max_val, vh):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            max_val: the normalization value
            vh: the SVD component needed for coil compression
        """

        # kspace is dimension [num_coils, size0, size1, 2]
        kspace = to_tensor(kspace)

        # Compress to virtual coils
        gt_img, gt_k = get_compressed(kspace, self.img_size, num_vcoils=self.num_vcoils, vh=vh)

        # Stack the coils and real and imaginary
        gt_img = to_tensor(gt_img).permute(2, 3, 0, 1).reshape(-1, self.img_size, self.img_size).unsqueeze(0)

        # Apply the mask
        if isinstance(self.accel_rate, list):
            choice = np.random.choice(len(self.accel_rate))
            accel = self.accel_rate[choice]
        else:
            accel = self.accel_rate

        mask = get_mask(accel=accel, size=self.img_size, mask_type=self.mask_type)
        masked_kspace = apply_mask(gt_k, mask)

        # Get the zf imgs
        masked_img = fastmri.ifft2c(masked_kspace).permute(0, 3, 1, 2).reshape(-1, self.img_size,
                                                                               self.img_size).unsqueeze(0)

        # Normalized based on the 95th percentile max value of the magnitude
        zf_img = masked_img / max_val
        gt_img = gt_img / max_val

        return zf_img, gt_img, mask




# Function to get the compressed image and kspace
def get_compressed(kspace: np.ndarray, img_size, num_vcoils = 8, vh = None):
    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace)
    
    compressed_imgs = []

    #Compress to 8 virtual coils and crop
    compressed_img = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img).transpose(1,2,0),
                                                   img_size, num_vcoils, vh)

        
    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(to_tensor(compressed_img).permute(2,0,1,3))
    
    
    return compressed_img, compressed_k






    

# Example usage
if __name__ == '__main__':


    kwargs = {
                'mri_type': 'brain',  # brain or knee
                'center_frac': 0.08,
                'accel_rate': 4,
                'img_size': 384,
                'challenge': "multicoil",
                'complex': True, # if singlecoil, specify magnitude or complex
                'scan_type': None,  # Knee: 'CORPD_FBK' Brain: 'AXT2'
                'mask_type': 'brain',  # Options :'s4', 'default', 'center_aug'
                'num_vcoils': 8,
                'acs_size': 32,  # 13 for knee, 32 for brain
                'slice_range': [0,8],  # [0, 8], None
                'augmented': False,
                'specific_label': None, #'Mass',
            }

    # Location of the dataset
    if kwargs['mri_type'] == 'brain':
        base_dir = "/storage/fastMRI_brain/data/"
    elif kwargs['mri_type'] == 'knee':
        base_dir = "/storage/fastMRI/data/"
    else:
        raise Exception("Please specify an mri_type in configs")
    
    data = FastMRIDataModule(base_dir, batch = 4, **kwargs)
    data.prepare_data()
    data.setup()
    #dataset = MulticoilDataset(dataset_dir, max_val_dir, img_size, mask_type)
    dataset = data.val
    for i in tqdm(range(len(dataset))):
    #i = 2388
        cond = dataset[i][0]
        img = dataset[i][1]
        mask = dataset[i][2]
        norm_val = dataset[i][3]
        label = dataset[i][-1]

        if label == 1:
            break

    viz.show_img(img.unsqueeze(0),rss=True)

    maps = network_utils.get_maps(cond.unsqueeze(0), data.acs, norm_val)
    mag_img = network_utils.get_magnitude(img.unsqueeze(0), maps, rss=False)
    viz.show_img(mag_img)
    

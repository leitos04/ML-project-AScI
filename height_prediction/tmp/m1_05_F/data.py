import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

class SPMDataset(Dataset):
    '''
    Pytorch dataset for STM+AFM data using HDF5 database.
    Arguments:
        hdf5_path: str. Path to HDF5 database file.
        mode: 'train', 'val', or 'test'. Which dataset to use.
        scan: 'afm' or 'stm'.
    '''
    def __init__(self, hdf5_path, mode='train', scan=None, height=None, transform=None, transform_Y=None, Y_channel=None):
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.scan = scan
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(f'mode should be one of "train", "val", or "test", but got {self.mode}')
        self.h = height

        self.transform = transform
        self.transform_Y = transform_Y
        self.Y_channel = Y_channel

    def __len__(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            length = len(f[self.mode]['X'])
        return length

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            dataset = f[self.mode]
            X = dataset['X'][idx]
            Y = dataset['Y'][idx] if 'Y' in dataset.keys() else []
            xyz = dataset['xyz'][idx]
            if self.scan == 'stm':
                X = X[0]
            if self.scan == 'afm':
                X = X[1]
            if isinstance(self.h, int) and self.scan:
                X = X[:, :, self.h]
            if isinstance(self.h, int) and not self.scan:
                X = X[:, :, :, self.h]
            if self.h == 'random':
                h = np.random.randint(X.shape[-1])
                X = X[:, :, h]

            X = X[None, :, :]
            h = 4.3 - h*0.1
            if self.Y_channel:
                Y = Y[self.Y_channel]
            if self.transform:
                X = self.transform(torch.from_numpy(X))
            if self.transform_Y:
                Y = self.transform_Y(torch.from_numpy(Y))

        return X, h

    def _unpad_xyz(self, xyz):
        return xyz[xyz[:,-1] > 0]
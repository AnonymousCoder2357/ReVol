"""
Utility functions for device configuration and data loading.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def to_device(gpu):
    """
    Select the appropriate computation device (CPU or CUDA).

    param gpu: GPU device index (e.g., 0, 1, ...), or None to use CPU.
    
    return: torch.device object representing either a specific GPU or CPU.
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


def to_loader(x, y, xm, ym, batch_size, shuffle=False):
    """
    Create a PyTorch DataLoader from given NumPy arrays, including mask information.

    param x: Input features (NumPy array).
    param y: Target values (NumPy array).
    param xm: Input mask, indicating valid x entries (NumPy array).
    param ym: Target mask, indicating valid y entries (NumPy array).
    param batch_size: Number of samples per batch.
    param shuffle: Whether to shuffle the dataset at every epoch.

    return: A DataLoader containing a TensorDataset of (x, y, xm, ym).
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    xm = torch.from_numpy(xm)
    ym = torch.from_numpy(ym)
    return DataLoader(TensorDataset(x, y, xm, ym), batch_size, shuffle)

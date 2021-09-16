"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        # Create two new keys in the "sample" dictionary, named "image" and "seg"
        # The values are 3D Torch Tensors with image and label data respectively. 
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices. Write code that stores the 2D slice data in the last 2 dimensions of the 3D Tensors. 
        # Tensor needs to be of shape [1, patch_size, patch_size]

        sample["image"] = torch.from_numpy(self.data[slc[0]]['image'][slc[1]][None, :]).type(torch.FloatTensor)
        sample["seg"] = torch.from_numpy(self.data[slc[0]]['seg'][slc[1]][None, :]).type(torch.LongTensor)
        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)

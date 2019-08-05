import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp


def make_dataset(data_path, taget):
    image_list = open(data_path).readlines()
    if taget:
      len_ = len(image_list)
      dataset = [(image_list[i].strip(), taget[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        dataset = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        dataset = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return dataset


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data_path, target=None, transform=None, target_transform=None, root_path=None):
        self.dataset = make_dataset(data_path, target)
        self.transform = transform
        self.target_transform = target_transform
        self.root_path = root_path

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.dataset[index]
        if self.root_path != None:
            image = Image.open(osp.join(self.root_path, path)).convert('RGB')
        else:
            image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.dataset)

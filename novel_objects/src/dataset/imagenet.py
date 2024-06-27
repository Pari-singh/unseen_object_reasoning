from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
import copy
from typing import Any, Callable, Optional, Tuple
import cv2

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets import ImageNet
from itertools import chain

import novel_objects.src.dataset.generate_heirarchy_imagenet as class_info
# import novel_objects.src.scripts.opts as opts


class ImageNetMini(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'imagenet'

    train_list = [
        'train_data_batch_1',
        'train_data_batch_2',
        'train_data_batch_3',
        'train_data_batch_4',
        'train_data_batch_5',
        'train_data_batch_6',
        'train_data_batch_7',
        'train_data_batch_8',
        'train_data_batch_9',
        'train_data_batch_10',
    ]

    test_list = [
        ['test_data_batch_5'],
    ]

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            debug_dataset_mini: bool = False,
            use_patch: bool = False,
            patch_res: int = 2
    ) -> None:

        super(ImageNetMini, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        if debug_dataset_mini:
            print("WARNING: ****LOADING MINI DATASET****")
            downloaded_list = downloaded_list[:2]

        self.data: Any = []
        self.all_data: Any = []
        self.targets = []
        self.all_targets = []

        self.combined_classes = {}
        for key, value in class_info.superclasses_64x64.items():
            self.combined_classes.update(value)
        valid_indices = list(self.combined_classes.keys())
        self.reindexed_classes = {i: list(self.combined_classes.keys())[i]
                                  for i in range(len(list(self.combined_classes.keys())))}
        self.reversed_reindexed_classes = {v: k for k, v in self.reindexed_classes.items()}

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.all_data.append(entry['data'])
                if 'labels' in entry:
                    self.all_targets.extend(entry['labels'])

        valid_targets = [i for i, x in enumerate(self.all_targets) if x in valid_indices]

        self.all_data = np.vstack(self.all_data).reshape(-1, 3, 64, 64)
        self.all_data = self.all_data.transpose((0, 2, 3, 1))  # convert to HWC


        # data & targets with valid indices
        self.data = self.all_data[valid_targets]
        self.targets = [self.reversed_reindexed_classes[self.all_targets[x]] for x in valid_targets]

        if debug_dataset_mini:
            self.data = self.data[:500]
            self.targets = self.targets[:500]

        self._load_meta()
        self.num_classes = len(self.classes)
        self.use_patch = use_patch
        self.patch_res = patch_res

    def _load_meta(self) -> None:

        self.classes = list(self.combined_classes.values())
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # img = cv2.resize(img, (64, 64))

        # target tensor
        if self.use_patch:
            target_tensor_ohe = np.zeros((self.num_classes, int(img.shape[0]/self.patch_res),
                                          int(img.shape[1]/self.patch_res)), dtype=np.int32)
            target_tensor = np.zeros((int(img.shape[0]/self.patch_res), int(img.shape[1]/self.patch_res)),
                                     dtype=np.int32)
        else:
            target_tensor_ohe = np.zeros((self.num_classes, img.shape[0], img.shape[1]), dtype=np.int32)
            target_tensor = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        raw_img = copy.deepcopy(img)
        raw_img = np.asarray(raw_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target_tensor_ohe[target, :, :] = 1
        target_tensor[:, :] = target
        target_tensor_ohe = torch.tensor(target_tensor_ohe)
        target_tensor = torch.tensor(target_tensor)

        return img, target_tensor, target_tensor_ohe, raw_img

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



class ImageNetKmeans(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'imagenet'

    train_list = [
        'train_data_batch_1',
        'train_data_batch_2',
        'train_data_batch_3',
        'train_data_batch_4',
        'train_data_batch_5',
        'train_data_batch_6',
        'train_data_batch_7',
        'train_data_batch_8',
        'train_data_batch_9',
        'train_data_batch_10',
    ]

    test_list = [
        ['test_data_batch_5'],
    ]

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            debug_dataset_mini: bool = False,
            use_patch: bool = False,
            patch_res: int = 2
    ) -> None:

        super(ImageNetKmeans, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        self.train = train  # training set or test set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        if debug_dataset_mini:
            print("WARNING: ****LOADING MINI DATASET****")
            downloaded_list = downloaded_list[:2]

        self.data: Any = []
        self.all_data: Any = []
        self.targets = []
        self.all_targets = []

        self.combined_classes = {}
        for key, value in class_info.superclasses_64x64.items():
            self.combined_classes.update(value)
        valid_indices = list(self.combined_classes.keys())
        self.reindexed_classes = {i: valid_indices[i]
                                  for i in range(len(valid_indices))}
        self.reversed_reindexed_classes = {v: k for k, v in self.reindexed_classes.items()}

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.all_data.append(entry['data'])
                if 'labels' in entry:
                    self.all_targets.extend(entry['labels'])

        valid_targets = [i for i, x in enumerate(self.all_targets) if x in valid_indices]

        self.all_data = np.vstack(self.all_data).reshape(-1, 3, 64, 64)
        self.all_data = self.all_data.transpose((0, 2, 3, 1))  # convert to HWC


        # data & targets with valid indices
        self.data = self.all_data[valid_targets]
        self.targets = [self.reversed_reindexed_classes[self.all_targets[x]] for x in valid_targets]

        if debug_dataset_mini:
            self.data = self.data[:500]
            self.targets = self.targets[:500]

        self._load_meta()
        self.num_classes = len(self.classes)

    def _load_meta(self) -> None:

        self.classes = list(self.combined_classes.values())
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # img = cv2.resize(img, (64, 64))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        raw_img = copy.deepcopy(img)
        raw_img = np.asarray(raw_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, raw_img

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
import copy
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive




class CIFAR10(VisionDataset):
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
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    hierarchy_metadata = {}
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            use_patch: bool = False,
            patch_res: int = 2
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.num_classes = len(self.classes)
        self.use_patch = use_patch
        self.patch_res = patch_res

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # target tensor
        if self.use_patch:
            target_tensor_ohe = np.zeros((self.num_classes, int(img.shape[0]/self.patch_res),
                                          int(img.shape[1]/self.patch_res)), dtype=np.int32)
            target_tensor = np.zeros((int(img.shape[0] / self.patch_res), int(img.shape[1] / self.patch_res)),
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


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class cifarCombined(VisionDataset):
    base_folder = ['cifar-10-batches-py', 'cifar-100-python']
    url = ["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",  "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"]
    filename = ["cifar-10-python.tar.gz", "cifar-100-python.tar.gz"]
    tgz_md5 = ['c58f30108f718f92721af3b95e74349a', 'eb9058c3a382ffc7106e4002c42a8d85']

    hierarchy_metadata = {}
    train_list = [[
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ], [['train', '16019d7e3df5f24257cddd939b257f8d'],]]

    test_list = [[
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ], [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],]]
    meta = [{
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }, {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }]


    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(cifarCombined, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.num_dataset = len(self.base_folder)

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        downloaded_list = []
        for i in range(self.num_dataset):
            downloaded_list.append(self.train_list[i] + self.test_list[i])

        self.classes = []
        self.classes_acc = []
        self._load_meta()
        self.data: Any = []
        self.data_dummy = []
        self.targets = []

        # now load the picked numpy arrays
        counter = {x:0 for x in self.classes}
        for iter in range(self.num_dataset):
            for file_name, checksum in downloaded_list[iter]:
                file_path = os.path.join(self.root, self.base_folder[iter], file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    if 'labels' in entry:
                        for i in range(len(entry['labels'])):
                            if counter[self.classes_acc[iter][entry['labels'][i]]] < 600:
                                self.targets.append(self.class_to_idx[self.classes_acc[iter][entry['labels'][i]]])
                                self.data.append(entry['data'][i])
                                counter[self.classes_acc[iter][entry['labels'][i]]] += 1

                        # self.targets.extend([self.class_to_idx[self.classes_acc[iter][entry['labels'][i]]]
                        #                      for i in range(len(entry['labels']))])
                        # self.targets.extend(entry['labels'])
                    else:
                        for i in range(len(entry['fine_labels'])):
                            if counter[self.classes_acc[iter][entry['fine_labels'][i]]] < 600:
                                self.targets.append(self.class_to_idx[self.classes_acc[iter][entry['fine_labels'][i]]])
                                self.data.append(entry['data'][i])
                                counter[self.classes_acc[iter][entry['fine_labels'][i]]] += 1

                        # self.targets.extend([self.class_to_idx[self.classes_acc[iter][entry['fine_labels'][i]]]
                        #                      for i in range(len(entry['fine_labels']))])

                    # self.data.append(entry['data'])
        # counter = {x:[] for x in self.classes}
        # for counter[x].append

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.num_classes = len(self.classes)


    def _load_meta(self) -> None:

        for iter in range(self.num_dataset):
            path = os.path.join(self.root, self.base_folder[iter], self.meta[iter]['filename'])
            if not check_integrity(path, self.meta[iter]['md5']):
                raise RuntimeError('Dataset metadata file not found or corrupted.' +
                                   ' You can use download=True to download it')
            with open(path, 'rb') as infile:
                data = pickle.load(infile, encoding='latin1')
                self.classes += data[self.meta[iter]['key']]
                self.classes_acc.append(data[self.meta[iter]['key']])
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        self.combined_classes = {v: k for k, v in self.class_to_idx.items()}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # target tensor
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

    def _check_integrity(self, iter) -> bool:
        root = self.root
        for fentry in (self.train_list[iter] + self.test_list[iter]):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder[iter], filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        for iter in range(self.num_dataset):
            if self._check_integrity(iter):
                print('Files already downloaded and verified')
            else:
                download_and_extract_archive(self.url[iter], self.root, filename=self.filename[iter], md5=self.tgz_md5[iter])

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



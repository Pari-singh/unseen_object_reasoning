import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import os
from copy import deepcopy

class FeatureVectorDataset(Dataset):

    def __init__(self, base_dataset, feature_root):

        """
        Dataset loads feature vectors instead of images
        :param base_dataset: Dataset from which images would come
        :param feature_root: Root directory of features

        feature root should contain numpy files
        """

        self.base_dataset = base_dataset
        # self.target_transform = deepcopy(base_dataset.target_transform)
        #
        # self.base_dataset.target_transform = None
        # self.base_dataset.transform = None

        self.feature_root = feature_root
        # self.feature_split = os.listdir(self.feature_root)

    def __getitem__(self, item):

        # Get meta info of this instance
        img, label, uq_idx, raw_img = self.base_dataset[item]
        # Find the numpy file in one of the splits
        numpy_file_path = os.path.join(self.feature_root, f'{label}', f'{uq_idx}.npy')
        if not os.path.isfile(numpy_file_path):
            print("Couldn't find features at ", numpy_file_path)
            feature_vector = []
        else:
            # Load feature vector
            feature_vector = torch.from_numpy(torch.load(numpy_file_path).squeeze()[1:, :])


        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        #     label = self.target_transform(label)

        return feature_vector, img, label, raw_img

    def __len__(self):
        return len(self.base_dataset)


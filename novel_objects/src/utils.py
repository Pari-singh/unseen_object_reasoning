import random
import matplotlib.pyplot as plt

## PyTorch
import torch
from torch import default_generator, randperm
# PyTorch Lightning
from torch._utils import _accumulate


def split_train_test(lengths, total_classes,
                     generator=default_generator):
    """dataset is the overall dataset=train+test
    Splitting based on class"""

    if sum(lengths) != len(total_classes.keys()):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    random_classes = randperm(sum(lengths), generator=generator).tolist()
    return [random_classes[offset - length: offset] for offset, length in zip(_accumulate(lengths), lengths)]


def visualization(train_dataset, trainval_dataset, train_split_map):
    display_images = []
    _, axs = plt.subplots(5, 5, figsize=(12, 12))
    axs = axs.flatten()
    for i in range(25):
        idx = random.randint(0, len(trainval_dataset.targets))
        img = trainval_dataset.data[idx]
        label = list(train_split_map.keys())[list(train_split_map.values()).index(trainval_dataset.targets[idx])]
        labelname = train_dataset.combined_classes[list(train_dataset.combined_classes.keys())[label]]
        axs[i].imshow(img)
        axs[i].set_title(labelname, loc='center')
    plt.show()


def get_train_images(num, train_dataset):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)
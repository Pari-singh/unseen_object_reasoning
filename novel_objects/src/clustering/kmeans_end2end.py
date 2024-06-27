import argparse
from tqdm import tqdm
import os
import numpy as np
import random

import torch
import torch.utils.data as data
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from novel_objects.src.dataset.cifar10_custom_kmeans import CustomCIFAR10 as CIFAR10
from novel_objects.src.dataset.feature_vector_dataset import FeatureVectorDataset
from faster_k_means_pytorch import K_Means

def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dataloader(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=10),
                                    transforms.RandomGrayscale(p=0.1), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=.5, hue=.3),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_transform, test_transform = None, None
    train_dataset = CIFAR10(root=args.dataset_path, train=True, transform=transform)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root=args.dataset_path, train=False)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(range(10)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    print("Feature vector datasets...")
    # Pass both train and test folder for feature root as the split doesn't make sense for our work - was done by Sagar guys

    train_dataset = FeatureVectorDataset(base_dataset=train_set, feature_root=args.save_dir)
    val_dataset = FeatureVectorDataset(base_dataset=val_set, feature_root=args.save_dir)
    test_dataset = FeatureVectorDataset(base_dataset=test_set, feature_root=args.save_dir)
    train_dataset.target_transform = target_transform
    val_dataset.target_transform = target_transform

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                   drop_last=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, num_workers=args.num_workers, batch_size=args.batch_size,
                                  shuffle=False)
    return train_loader, val_loader, test_loader


def kmeans_batchwise(loader, args, K=None):

    all_feats = []
    raw_imgs = []
    targets = np.array([])

    for batch_idx, (feats, label, _, raw_img) in enumerate(tqdm(loader)):
        feats = feats.to(device)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        raw_imgs.append(raw_img.cpu().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    all_feats = np.concatenate(all_feats)
    raw_imgs = np.concatenate(raw_imgs)

    # if args.kmeans_cosine:
    #     C =


    print('Fitting K-Means...')
    kmeans = K_Means(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                     n_init=args.k_means_init, random_state=None, n_jobs=args.n_jobs, pairwise_batch_size=10,
                     mode=None)
    # TODO : figure out if labels should affect kmeans
    # all_feats, targets = (torch.from_numpy(x).to(device) for
    #                       x in (all_feats, targets))

    # for dot product"
    af = all_feats.reshape(-1, all_feats.shape[-1])
    kmeans.fit(af)
    all_preds = kmeans.labels_.cpu().numpy()

    # TODO: Add evaluation part
    return all_preds, kmeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='cluster'
    )
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--K', default=25, type=int, help='Set manually to run with custom K')
    parser.add_argument('--root_data_dir', type=str, default='novel_objects/data/')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, scars')
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_path', type=str, default='../../data')
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=None)
    args = parser.parse_args()
    seed_torch(0)
    args.save_dir = os.path.join(args.root_data_dir, f'{args.model_name}_{args.dataset_name}')

    device = torch.device('cuda:0')
    args.device = device
    print(args)
    # ----------------------
    # DATASETS
    # ----------------------
    train_loader, val_loader, test_loader = dataloader(args)

    kmeans_batchwise(train_loader, args, K=args.K)



def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)
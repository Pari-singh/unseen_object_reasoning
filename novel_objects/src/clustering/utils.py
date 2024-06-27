import torch
import random
import os
import numpy as np
import time
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import re, seaborn as sns
from matplotlib.colors import ListedColormap


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def torch_distance(data1, data2, batch_size):
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    i = 0
    closest_cluster = torch.zeros(data1.shape[0])
    closest_dist = torch.zeros(data1.shape[0])
    start_time = time.time()
    while i < data1.shape[0]:
        print(i)
        if (i + batch_size < data1.shape[0]):
            dis_batch = (A[i:i + batch_size] - B) ** 2
            dis_batch = dis_batch.sum(dim=-1)
            dis_batch_indices = torch.min(dis_batch, dim=-1).indices
            dis_batch_values = torch.min(dis_batch, dim=-1).values
            closest_cluster[i:i + batch_size] = dis_batch_indices
            closest_dist[i:i + batch_size] = dis_batch_values
            i = i + batch_size
            #  torch.cuda.empty_cache()
        elif (i + batch_size >= data1.shape[0]):
            dis_batch = (A[i:] - B) ** 2
            dis_batch = dis_batch.sum(dim=-1)
            dis_batch_indices = torch.min(dis_batch, dim=-1).indices
            dis_batch_values = torch.min(dis_batch, dim=-1).values
            closest_cluster[i:] = dis_batch_indices
            closest_dist[i:] = dis_batch_values
            #  torch.cuda.empty_cache()
            break
    print(time.time() - start_time)
    return closest_cluster, closest_dist



def pca(all_feats_reshaped, all_preds):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(all_feats_reshaped)
    pca_one = pca_result[:, 0]
    pca_two = pca_result[:, 1]
    pca_three = pca_result[:, 2]
    classes = list(set(all_preds))
    for i in range(len(classes)//7):
        sample_idx = []
        classes_to_sample_from = classes[i*7: (i+1)*7]
        random_class = random.sample(classes_to_sample_from, 7)
        for clas in random_class:
            sample_idx += list(np.where(all_preds==clas)[0])
        pca_one_sampled = [pca_one[idx] for idx in sample_idx]
        pca_two_sampled = [pca_two[idx] for idx in sample_idx]
        pca_three_sampled = [pca_three[idx] for idx in sample_idx]
        y_sampled = [all_preds[idx] for idx in sample_idx]
        fig = plt.figure(figsize=(16, 10))
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        color_map = plt.get_cmap('viridis')
        sc = ax.scatter(pca_one_sampled, pca_two_sampled, pca_three_sampled, s=40, c=y_sampled, marker='o', cmap=color_map, alpha=1)

        ax.set_xlabel('pca_one')
        ax.set_ylabel('pca_two')
        ax.set_zlabel('pca_three')

        # legend
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

        # save
        plt.savefig(f'TSNE_{i}th_7_classes', bbox_inches='tight')

def pca_per_class(all_feats, all_preds, targets):
    pca = PCA(n_components=3)
    classes = list(set(targets))
    j=0
    while True:
        sample_class = np.random.choice(classes, size=10, replace=False)
        idx = [i for i in range(len(targets)) if targets[i] in sample_class]
        all_feats_sample = all_feats[idx].reshape(-1, all_feats.shape[-1])
        target_sample = targets[idx]
        all_preds_sample = all_preds.reshape(*all_feats.shape[:-1])[idx].reshape(-1)
        pca_result = pca.fit_transform(all_feats_sample)
        pca_one = pca_result[:, 0]
        pca_two = pca_result[:, 1]
        pca_three = pca_result[:, 2]

        # First plot
        y = np.array([target_sample[idx // (all_feats.shape[1] ** 2)] for idx
                      in range(all_feats_sample.shape[0])])
        plt.close()
        fig = plt.figure(figsize=(16, 10))
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        color_map = plt.get_cmap('spring')
        sc = ax.scatter(pca_one, pca_two, pca_three, s=40, c=y, marker='o', cmap=color_map, alpha=1)
        ax.set_xlabel('pca_one')
        ax.set_ylabel('pca_two')
        ax.set_zlabel('pca_three')
        # legend
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f'TSNE_classwise_{j}.png', bbox_inches='tight')

        # Second plot
        y = all_preds_sample
        plt.close()
        fig = plt.figure(figsize=(16, 10))
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        color_map = plt.get_cmap('spring')
        sc = ax.scatter(pca_one, pca_two, pca_three, s=40, c=y, marker='o', cmap=color_map, alpha=1)
        ax.set_xlabel('pca_one')
        ax.set_ylabel('pca_two')
        ax.set_zlabel('pca_three')
        # legend
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
        plt.savefig(f'TSNE_clusterwise_{j}.png', bbox_inches='tight')
        plt.close()
        print(f"{j} Done")
        j += 1
        if j>15:
            break




def posemb_sincos_2d(patches_shape, temperature=10000, dtype=torch.float32):

    _, h, w, dim, dtype = *patches_shape, dtype

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

def visualization_eval(args, vis_superclass, raw_imgs):
    plt.ioff()
    for superclass, classes in enumerate(vis_superclass):
        fig, axs = plt.subplots(10, 10, figsize=(25,25))
        axs = axs.flatten()
        size = len(classes) if len(classes)<100 else 100
        classes_rand = np.random.choice(classes, size, replace=False)
        for i, classes_randl in enumerate(classes_rand):
            axs[i].imshow(raw_imgs[classes_randl])
        folder_name = 'eval_clusters_' + (args.saved_kmeans.split('_')[-1]).split('.')[0]
        os.makedirs(folder_name, exist_ok=True)
        fig.savefig(f'{folder_name}/{superclass}.png')


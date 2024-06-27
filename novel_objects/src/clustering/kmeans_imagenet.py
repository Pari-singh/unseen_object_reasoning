import argparse
import pdb

from tqdm import tqdm
import os
import numpy as np
from numpy import save, load
import json
from collections import Counter
import scipy.stats as ss
import kmeans_pytorch as tkmeans
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from torch import default_generator, randperm
from torch._utils import _accumulate
import torchvision.transforms as T
from torch.utils.data import DataLoader

from novel_objects.src.dataset.imagenet import ImageNetKmeans as ImageNetMini
import novel_objects.src.dataset.generate_heirarchy_imagenet as ghi
from novel_objects.src.dataset.feature_vector_dataset import FeatureVectorDataset
from novel_objects.src.clustering.eval_imagenet import Evaluation
from novel_objects.src.clustering.evaluation import Evaluation as crossdataEval
from novel_objects.src.clustering.utils import seed_torch, torch_distance, pca
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from novel_objects.src.models import vision_transformer as vits
import copy
import novel_objects.src as src


class ClassKMeans:

    def __init__(self, args):
        self.num_train_classes = 0
        self.num_test_classes = 0
        self.args = args

    def extract_features_dino(self, model, loader, save_dir):

        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):

                images, labels, idxs = batch[:3]
                images = images.to(device)
                # Save features
                for i, t, uq in zip(images, labels, idxs):
                    t = t.item()
                    uq = uq.item()
                    feat, _ = model.get_last_selfattention(i.unsqueeze(0))
                    save_folder = os.path.join(save_dir, f'{t}')
                    save_path = os.path.join(save_folder, f'{uq}.npy')
                    os.makedirs(save_folder, exist_ok=True)
                    torch.save(feat.detach().cpu().numpy(), save_path)

    def visualization(self, cluster_dct, clusters, raw_imgs, pool_size=1, num_patches=13):
        plt.ioff()
        args.interpolation = 3
        args.crop_pct = 0.875
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 224
        interpolation = args.interpolation
        crop_pct = args.crop_pct
        transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()])

        for j, cluster in enumerate(clusters):
            fig, axs = plt.subplots(7,7, figsize=(15, 15))
            axs = axs.flatten()
            val = cluster_dct[j]
            val.sort(key=lambda y: y[1])
            if len(val) < 50:
                continue
            print(len(val))
            for i in range(49):
                idx = val[i][0]
                image = raw_imgs[idx // (num_patches ** 2)]
                patch = idx // len(raw_imgs)
                row, col = patch // num_patches, patch % num_patches
                t_img = transform(Image.fromarray(image)).permute(1, 2, 0).numpy()
                patch_size_begin = 16 * args.stride
                patch_size_end = 16 * pool_size
                img_patch = t_img[(row * patch_size_begin):((row * patch_size_begin)+patch_size_end),
                            (col * patch_size_begin):((col * patch_size_begin)+patch_size_end)]
                label = int(val[i][2])
                labelname = ghi.classnames[ghi.index_map[label]]
                axs[i].imshow(img_patch)
                axs[i].set_title(labelname, loc='center')
                if i == len(val) - 1:
                    break
            folder_name = 'imagenet'+(args.saved_kmeans.split('_')[-1]).split('.')[0]
            os.makedirs(folder_name, exist_ok=True)
            fig.savefig(f'{folder_name}/cluster_{j}.png')
            if j>=50:
                break
        plt.close()

    def imagenet_visualization(self, trainval_dataset, train_split_map, train_dataset):
        _, axs = plt.subplots(5, 5, figsize=(12, 12))
        axs = axs.flatten()
        for i in range(25):
            idx = random.randint(0, len(trainval_dataset.targets))
            img = trainval_dataset.data[idx]
            label = list(train_split_map.keys())[
                list(train_split_map.values()).index(trainval_dataset.targets[idx])]
            labelname = train_dataset.combined_classes[list(train_dataset.combined_classes.keys())[label]]
            axs[i].imshow(img)
            axs[i].set_title(labelname, loc='center')
        plt.savefig("imagenet.png")

    def split_train_test(self, lengths, total_classes,
                         generator=default_generator):
        """dataset is the overall dataset=train+test. Splitting based on class"""

        if sum(lengths) != len(total_classes.keys()):  # type: ignore[arg-type]
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        random_classes = randperm(sum(lengths), generator=generator).tolist()
        return [random_classes[offset - length: offset] for offset, length in
                zip(_accumulate(lengths), lengths)]

    def dataloader(self):

        # Transform from the Dino paper:
        args.interpolation = 3
        args.crop_pct = 0.875
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        image_size = 224
        interpolation = args.interpolation
        crop_pct = args.crop_pct
        transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        print('Loading model...')
        # ----------------------
        # MODEL
        # ----------------------
        pretrain_path = os.path.join(self.args.pretrain_path)
        model = vits.__dict__['vit_base']()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        DATASET_PATH = src.top_dir() + "/data"
        assert os.path.exists(DATASET_PATH)

        train_dataset = ImageNetMini(root=DATASET_PATH, train=True, transform=transform, download=True)


        train_test_classes = [0.75, 0.25]
        total_classes_train = len(train_dataset.classes)
        self.num_train_classes = np.uint32(np.ceil(total_classes_train * train_test_classes[0]))  # TODO: DO NOT HARDCODE
        self.num_test_classes = len(train_dataset.classes) - self.num_train_classes
        train_classes, test_classes = self.split_train_test(lengths=[self.num_train_classes, self.num_test_classes],
                                                       total_classes=train_dataset.combined_classes)

        train_idx = [x for x in range(len(train_dataset.targets)) if train_dataset.targets[x] in train_classes]
        test_idx = [x for x in range(len(train_dataset.targets)) if train_dataset.targets[x] in test_classes]
        trainval_dataset = copy.deepcopy(train_dataset)
        test_dataset = copy.deepcopy(train_dataset)
        # train_split_map = {list(set(trainval_dataset.targets))[x]: x for x in range(len(train_classes))}

        trainval_dataset.data, trainval_dataset.targets, trainval_dataset.num_classes = \
            trainval_dataset.data[train_idx], [trainval_dataset.targets[x] for x in train_idx], \
            len(train_classes)
        test_dataset.data, test_dataset.targets, test_dataset.num_classes = \
            test_dataset.data[test_idx], [test_dataset.targets[x] for x in test_idx], len(test_classes)

        num_classes = trainval_dataset.num_classes

        # Visualize for confirmation
        # self.imagenet_visualization(trainval_dataset)

        # Set target transforms:
        target_transform_dict = {}
        for i, cls in enumerate(range(len(train_classes)+len(test_classes))):
            target_transform_dict[cls] = i
        target_transform = lambda x: target_transform_dict[x]

        train_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)

        # # ----------------------
        # # EXTRACT FEATURES
        # # ----------------------
        if args.extract_features:
            # Extract train features
            train_save_dir = os.path.join('/'.join(args.save_dir.split('/')[:-1]), 'train')
            print('Extracting features from train split...')
            self.extract_features_dino(model=model, loader=train_loader, save_dir=train_save_dir)

            test_save_dir = os.path.join('/'.join(args.save_dir.split('/')[:-1]), 'test')
            print('Extracting features from test split...')
            self.extract_features_dino(model=model, loader=test_loader, save_dir=test_save_dir)

        trainval_dataset = FeatureVectorDataset(base_dataset=trainval_dataset, feature_root=args.save_dir)
        test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=args.save_dir)
        trainval_dataset.target_transform = target_transform


        train_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, shuffle=False,
                                       drop_last=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)

        return train_loader, test_loader

    def posemb_sincos_2d(self, patches_shape, temperature=10000, dtype=torch.float32):
        _, h, w, dim, dtype = *patches_shape, dtype

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1. / (temperature ** omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)


    def kmeans_classwise(self, loader, K=None):
        all_feats = np.array([])
        all_feats_orig = torch.tensor([])
        raw_imgs = []
        targets = np.array([])
        kernel_size = args.maxpool_size
        stride = (1, 1)
        pooling = torch.nn.MaxPool2d(kernel_size, stride)
        pool_output = torch.tensor([])
        for batch_idx, (feats, img, label, raw_img) in enumerate(tqdm(loader)):
            # feats = feats.to(device)
            feats = feats.reshape(-1, int(np.sqrt(feats.shape[1])),
                                  int(np.sqrt(feats.shape[1])), feats.shape[-1])
            feats = feats.permute(0, 3, 1, 2)
            pool_output = torch.cat((pool_output, (pooling(feats))), 0)
            targets = np.append(targets, label.numpy())
            raw_imgs.append(raw_img.numpy())
        all_feats = pool_output.permute(0, 2, 3, 1).numpy()
        # all_feats_orig = all_feats_orig.numpy()

        raw_imgs = np.concatenate(raw_imgs)

        if args.position_wts:
            posemb = self.posemb_sincos_2d(all_feats.shape)
            posemb_reshaped = posemb.reshape(*all_feats.shape[1:])
            all_feats = all_feats + args.position_wts*posemb_reshaped.numpy()

        # for dot product"
        all_feats_reshaped = all_feats.reshape(-1, all_feats.shape[-1])

        if args.use_saliency:
            features = torch.from_numpy(all_feats_orig)
            salient_feats = torch.sum(abs(features), dim=3)
            salient_feats = salient_feats / \
                            torch.amax(salient_feats, dim=(1,2)).view(salient_feats.size(0), 1,1)
            f = T.Resize(all_feats.shape[1], interpolation=T.InterpolationMode.BICUBIC)
            salient_feats_reshaped = f(salient_feats)
            all_salient_feats = salient_feats_reshaped.reshape(-1).numpy()

        # -----------------------
        # K-MEANS
        # -----------------------

        if os.path.isfile(os.path.join('imagenet', args.saved_kmeans)):
            kmeans = pickle.load(open(os.path.join('imagenet', args.saved_kmeans), 'rb'))
        elif args.full_kmeans:
            print('Fitting K-Means...')
            cluster_ids_x, cluster_centers = tkmeans.kmeans(X=torch.from_numpy(all_feats_reshaped),
                                                    num_clusters=K, distance='euclidean')
        else:
            # ssd= []
            for K in [K]:
                kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=6000, max_iter=5)
                for k in range(27):
                    num_samples = (6000 * all_feats.shape[1] * all_feats.shape[2])
                    rand_list = list(np.random.permutation(np.arange(k * num_samples, (k + 1) * num_samples)))
                    kmeans = kmeans.partial_fit(all_feats_reshaped[rand_list, :])
                print(f'Adding {K}')
            #     ssd.append(kmeans.inertia_)
            # plt.plot([200, 500, 700, 900, 1200, 1500, 1700, 2000, 2300], ssd, 'ro-')
            # plt.xlabel('k')
            # plt.ylabel('Sum_of_squared_distances')
            # plt.title('Elbow Method For Optimal k')
            # plt.show()
            # plt.close()
            os.makedirs('imagenet', exist_ok=True)
            pickle.dump(kmeans, open(os.path.join('imagenet', args.saved_kmeans), 'wb'))

        clusters = kmeans.cluster_centers_
        if os.path.isfile(os.path.join('imagenet', args.cluster_dct)) \
                and os.path.isfile(os.path.join('imagenet', args.all_preds)):
            with open(os.path.join('imagenet', args.cluster_dct), 'rb') as f:
                cluster_dct = pickle.load(f)
            all_preds = np.load(os.path.join('imagenet', args.all_preds))
        else:

            cluster_dct = {}
            all_preds, dist = torch_distance(torch.from_numpy(all_feats_reshaped),
                                             torch.from_numpy(clusters), args.batch_size)
            os.makedirs('imagenet', exist_ok=True)
            np.save(os.path.join('imagenet', args.all_preds),
                    all_preds.numpy())
            for i, data in enumerate(all_preds):
                label = targets[i // (all_feats.shape[1] * all_feats.shape[2])]
                cluster_dct.setdefault(int(data), []).append((i, float(dist[i]), label))
            with open(os.path.join('imagenet', args.cluster_dct), 'wb') as f:
                pickle.dump(cluster_dct, f)

            # check with the saved one or save this one!



        #------------------------------
        # Visualization
        #------------------------------
        # self.visualization(cluster_dct, clusters, raw_imgs, pool_size=args.maxpool_size,
        #                    num_patches=all_feats.shape[1])


        # pca(all_feats_reshaped, all_preds)


        if os.path.isfile(os.path.join('imagenet', args.cluster_composition)):
            cluster_composition = np.load(os.path.join('imagenet', args.cluster_composition), allow_pickle=True)
        else:
            cluster_composition = [[0]*168 for _ in range(args.K)]
            cluster_lst = []
            for cluster in range(len(clusters)):
                val = np.array(cluster_dct[cluster])
                val_id = val[:, 2].astype(int)
                weights = np.ones_like(val_id)
                cluster_lst.append([cluster, len(val_id)])
                if args.use_saliency:
                    val_idx = val[:, 0].astype(int)
                    print(val_idx.max())
                    weights = all_salient_feats[val_idx]
                cluster_composition[cluster] = np.bincount(val_id, weights=weights, minlength=168) /\
                                               np.bincount(val_id, weights=weights, minlength=168).sum()

            ## Turn this on if performing max over entire cluster.
            # cluster_composition = np.array(cluster_composition)/np.array(cluster_composition).max()
            os.makedirs('imagenet', exist_ok=True)
            np.save(os.path.join('imagenet', args.cluster_composition),
                    np.array(cluster_composition))
            plt.close()
            plt.hist(np.array(cluster_lst)[:, 1], bins=np.array(cluster_lst)[:, 0])
            # plt.savefig("histimagenet_1k.png")
            # plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='cluster'
    )
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--K', default=800, type=int, help='Set manually to run with custom K')
    parser.add_argument('--root_data_dir', type=str, default='novel_objects/data/')
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='options: cifar10, cifar100, scars')
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--pretrain_path', type=str, default='novel_objects/src/saved_models/dino_vitbase16_pretrain.pth')
    parser.add_argument('--saved_kmeans', type=str, default='kmeans_cifar100_800cluster7.pkl')
    parser.add_argument('--cluster_composition', type=str, default='cluster_composition_800cluster7.npy')
    parser.add_argument('--cluster_dct', type=str, default='cluster_dct_800cluster7.pkl')
    parser.add_argument('--all_preds', type=str, default='all_preds_800cluster7.npy')
    parser.add_argument('--dataset_path', type=str, default='../../data')
    parser.add_argument('--max_kmeans_iter', type=int, default=300)
    parser.add_argument('--k_means_init', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=None)
    parser.add_argument('--maxpool_size', type=int, default=7)
    parser.add_argument('--averagepool_size', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--use_saliency', type=bool, default=False)
    parser.add_argument('--position_wts', type=float, default=0)
    parser.add_argument('--use_dominant_class_init', type=bool, default=False)
    parser.add_argument('--extract_features', type=bool, default=False)
    parser.add_argument('--crossdata_eval', type=bool, default=False)
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--inference', type=bool, default=False)
    parser.add_argument('--full_kmeans', type=bool, default=False)
    parser.add_argument('--inference_input_path', type=str, default='../../data/inference.png')
    parser.add_argument('--inference_feature_path', type=str, default=None)
    parser.add_argument('--DATASET_PATH', type=str, default="../../data")
    args = parser.parse_args()
    seed_torch(0)
    args.save_dir = os.path.join(args.root_data_dir, f'{args.model_name}_{args.dataset_name}', 'test') \
        if args.evaluation or args.crossdata_eval else os.path.join(args.root_data_dir, f'{args.model_name}_{args.dataset_name}', 'train')

    device = torch.device('cuda:0')
    args.device = device
    print(args)

    # ----------------------
    # DATASETS
    # ----------------------
    obj = ClassKMeans(args=args)

    train_loader, test_loader= obj.dataloader()


    if args.crossdata_eval:
        eval_obj = crossdataEval(args=args)
        eval_obj.eval(test_loader)
    elif args.evaluation or args.inference:
        eval_obj = Evaluation(args=args)
        if args.evaluation:
            eval_obj.eval(test_loader)
        else:
            eval_obj.inference()
    else:
        obj.kmeans_classwise(train_loader, K=args.K)


    print("Done")
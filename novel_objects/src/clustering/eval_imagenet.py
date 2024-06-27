from tqdm import tqdm
import os
import numpy as np
import time
import pickle
from PIL import Image
from scipy import misc

from novel_objects.src.clustering.utils import seed_torch, torch_distance, posemb_sincos_2d
from novel_objects.src.models import vision_transformer as vits
import novel_objects.src.dataset.generate_heirarchy_imagenet as ghi

import torch
import torchvision.transforms as T

class Evaluation:
    def __init__(self, args):
        self.args = args

        self.kmeans = pickle.load(open(os.path.join('imagenet', args.saved_kmeans), 'rb'))
        self.cluster_centers = self.kmeans.cluster_centers_

    def eval(self, inference_loader):
        cluster_composition = np.load(os.path.join('imagenet', self.args.cluster_composition), allow_pickle=True)
        raw_imgs = []
        targets = np.array([])


        if self.args.maxpool_size:
            all_feats = np.array([])
            all_feats_orig = torch.tensor([])
            kernel_size = self.args.maxpool_size
            stride = (1,1)
            pooling = torch.nn.MaxPool2d(kernel_size, stride)
            pool_output = torch.tensor([])
            for batch_idx, (feats, img, label, raw_img) in enumerate(tqdm(inference_loader)):
                # feats = feats.to(device)
                # feats = torch.nn.functional.normalize(feats, dim=-1)
                feats = feats.reshape(-1, int(np.sqrt(feats.shape[1])),
                                      int(np.sqrt(feats.shape[1])), feats.shape[-1])
                # all_feats_orig = torch.cat((all_feats_orig, feats), 0)
                feats = feats.permute(0, 3, 1, 2)
                pool_output = torch.cat((pool_output, (pooling(feats))), 0)
                # all_feats.append(feats.cpu().numpy())
                targets = np.append(targets, label.numpy())
                raw_imgs.append(raw_img.numpy())
            all_feats = pool_output.permute(0, 2,3,1).numpy()
            # all_feats_orig = all_feats_orig.numpy()
            if self.args.use_saliency:
                features = torch.from_numpy(all_feats_orig)
                salient_feats = torch.sum(abs(features), dim=3)
                salient_feats = salient_feats / \
                            torch.amax(salient_feats, dim=(1, 2)).view(salient_feats.size(0), 1, 1)
                f = T.Resize(all_feats.shape[1], interpolation=T.InterpolationMode.BICUBIC)
                salient_feats_reshaped = f(salient_feats)
                all_salient_feats = salient_feats_reshaped.reshape(-1).numpy()


        raw_imgs = np.concatenate(raw_imgs)
        if self.args.position_wts:
            posemb = posemb_sincos_2d(all_feats.shape)
            posemb_reshaped = posemb.reshape(*all_feats.shape[1:])
            all_feats = all_feats + self.args.position_wts * posemb_reshaped.numpy()

        all_feats_reshaped = all_feats.reshape(-1, all_feats.shape[-1])
        all_preds, dist = torch_distance(torch.from_numpy(all_feats_reshaped),
                                   torch.from_numpy(self.cluster_centers), self.args.batch_size)
        all_preds_distribution = torch.reshape(all_preds, (all_feats.shape[:-1]))

        top_1_accuracy = 0.0
        top_2_accuracy = 0.0
        top_3_accuracy = 0.0
        exits_accuracy = 0.0

        for i, instance in enumerate(all_preds_distribution):
            instance_class = np.array([0.0 for _ in range(168)])
            ground_truth_superclass = ghi.superclasses_map[ghi.class_to_superclass[ghi.index_map[targets[i]]]]
            instance_size = instance.shape[0]*instance.shape[1]
            for patch in range(0, instance_size):
                idx = i*instance_size + patch
                print(i, patch, idx)
                if self.args.use_saliency:
                    instance_class += cluster_composition[int
                    (all_preds[(i*(instance.shape[0]*instance.shape[1]))+patch])] * \
                                      all_salient_feats[idx]
                else:
                    instance_class += cluster_composition[int
                    (all_preds[(i * (instance.shape[0] * instance.shape[1])) + patch])]
            instance_class /= (instance.shape[0]*instance.shape[1])
            superclass_list = np.array([0.0 for _ in range(len(ghi.superclasses_map))])
            for k, score in enumerate(instance_class):
                superclass_list[ghi.superclasses_map[ghi.class_to_superclass[ghi.index_map[k]]]] += score
            superclass_list_arg = np.argsort(superclass_list)[::-1]
            if superclass_list_arg[0] == ground_truth_superclass:
                top_1_accuracy += 1
            if ground_truth_superclass in superclass_list_arg[:2]:
                top_2_accuracy += 1
            if ground_truth_superclass in superclass_list_arg[:3]:
                top_3_accuracy += 1
            if ground_truth_superclass in superclass_list_arg:
                exits_accuracy += 1

        top_1_accuracy_mean = top_1_accuracy / len(all_feats)
        top_2_accuracy_mean = top_2_accuracy / len(all_feats)
        top_3_accuracy_mean = top_3_accuracy / len(all_feats)
        exits_accuracy_mean = exits_accuracy / len(all_feats)


        print("------Results----------")
        print(top_1_accuracy, top_2_accuracy, top_3_accuracy)
        print(f"top 1 accuracy {top_1_accuracy_mean:.4f}  |  top 2 accuracy {top_2_accuracy_mean:.4f} "
              f"|  top 3 accuracy {top_3_accuracy_mean:.4f}")
        print("Done")

    def inference(self):
        print('Loading model...')
        # ----------------------
        # MODEL
        # ----------------------
        input_path = self.args.inference_input_path
        feature_path = self.args.inference_feature_path

        pretrain_path = os.path.join(self.args.pretrain_path)
        model = vits.__dict__['vit_base']()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        if os.path.isdir(input_path):
            ip_files = os.listdir(input_path)
            for file in ip_files:
                file_path = os.path.join(input_path, file)
                img = misc.imread(file_path)
                feat, _ = model.get_last_selfattention(img.unsqueeze(0))

        elif os.path.isfile(input_path) and input_path.endswith(('jpg', 'png')):
            img = misc.imread(input_path)
            feat, _ = model.get_last_selfattention(img.unsqueeze(0))
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import pdb
import pandas as pd
import seaborn as sn
import copy
import random

import torch
import torch.utils.data as data
from torchvision import transforms
import novel_objects.src.utils as utils
import novel_objects.src.network as network
from novel_objects.src.opts import opts as opts
from novel_objects.src.dataset.imagenet import ImageNetMini as data_source
from novel_objects.src.network import Encoder, Decoder
from novel_objects.src.network_res50 import UNetWithResnet50Encoder, UpBlockForUNetWithResNet50, ConvBlock, Bridge
from novel_objects.src.network import Autoencoder as default_autoencoder
from novel_objects.src.network_res50 import Autoencoder as res50_autoencoder


def init():
    random.seed(42)
    matplotlib.rcParams['lines.linewidth'] = 2.0
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    return device


def eval(opts, debug_data=False, debug_visualize=True, device=None):

    if not device:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    exp_id = opts.eval_exp_id
    latent_dim = opts.latent_dim
    ckpt_file_name = opts.eval_ckpt_file

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_dataset = data_source(root=opts.dataset_path, train=True, transform=transform,
                                download=True, debug_dataset_mini=debug_data)

    # create train & test class split - create disjoint train & test sets
    train_test_classes = [opts.train_class_split, opts.test_class_split]
    total_classes_train = len(train_dataset.classes)
    num_train_classes = np.uint32(np.ceil(total_classes_train * train_test_classes[0]))  # TODO: DO NOT HARDCODE
    num_test_classes = len(train_dataset.classes) - num_train_classes
    train_classes, test_classes = utils.split_train_test(lengths=[num_train_classes, num_test_classes],
                                                         total_classes=train_dataset.combined_classes)

    # create train & val split - from trainval dataset (does not use test)
    train_idx = [x for x in range(len(train_dataset.targets)) if train_dataset.targets[x] in train_classes]
    test_idx = [x for x in range(len(train_dataset.targets)) if train_dataset.targets[x] in test_classes]
    split_proportion = {'train': opts.train_data_split, 'val': opts.test_data_split}
    train_split = int(len(train_idx) * split_proportion['train'])
    val_split = len(train_idx) - int(len(train_idx) * split_proportion['train'])
    trainval_dataset = copy.deepcopy(train_dataset)
    test_dataset = copy.deepcopy(train_dataset)
    trainval_dataset.targets = [trainval_dataset.targets[x] for x in train_idx]
    train_split_map = {list(set(trainval_dataset.targets))[x]: x for x in range(len(train_classes))}

    # overwrite original object with split data
    trainval_dataset.data, trainval_dataset.targets, trainval_dataset.num_classes = \
        trainval_dataset.data[train_idx], [train_split_map[x] for x in trainval_dataset.targets], \
        len(train_classes)

    train_set, val_set = torch.utils.data.random_split(trainval_dataset, [train_split, val_split])

    # create val loaders
    val_loader = data.DataLoader(val_set, batch_size=opts.batch_size, shuffle=False, drop_last=False, num_workers=4)
    class_names_eval = [trainval_dataset.classes[x] for x in train_classes]

    ckpt_path = 'saved_models/' + opts.dataset + '_' + str(latent_dim) + '/lightning_logs/version_' + str(exp_id) + '/checkpoints'
    ckpt = os.path.join(opts.project_home, ckpt_path, ckpt_file_name)

    if opts.network == 'res50':

        model = res50_autoencoder(use_mode_loss=True,
                                  mode_loss_weights=[opts.loss_w1, opts.loss_w2],
                                  num_classes=len(train_classes), opts=opts)

    elif opts.network == 'default':
        model = default_autoencoder(base_channel_size=32, latent_dim=latent_dim, use_mode_loss=True,
                                    mode_loss_weights=[opts.loss_w1, opts.loss_w2], width=opts.img_width, height=opts.img_height,
                                    num_classes=len(train_classes), opts=opts)

    try:
        model.load_from_checkpoint(ckpt)
    except:
        raise ValueError("Loading model from checkpoint failed")

    confusion_matrix = np.zeros((len(class_names_eval), len(class_names_eval), 3), dtype=np.int32)
    for i in range(len(val_loader.dataset.indices)):
        batch = val_loader.dataset.__getitem__(i)
        x = batch[0][None, :]
        # raw_img = batch[3]
        sample_class = np.uint32(batch[1].numpy().mean())
        z = model.encoder(x)
        x_hat = model.decoder(z)
        x_hat = np.squeeze(x_hat.detach().cpu().numpy())
        predicted_class = np.argmax(np.sum(np.sum(x_hat, 1), 1))
        predicted_class_two = np.argsort(np.sum(np.sum(x_hat, 1), 1))[-2]
        predicted_class_three = np.argsort(np.sum(np.sum(x_hat, 1), 1))[-3]
        confusion_matrix[predicted_class, sample_class, 0] += 1
        confusion_matrix[predicted_class_two, sample_class, 1] += 1
        confusion_matrix[predicted_class_three, sample_class, 2] += 1
        print(i,  ' /', len(val_loader.dataset.indices))

    figures_path = os.path.join(opts.project_home, ckpt_path.rsplit('/', 1)[0], 'figures')
    os.makedirs(figures_path, exist_ok=True)
    for i in range(3):
        pd.set_option('display.float_format', '{:.2f}'.format)
        df_cm = pd.DataFrame(confusion_matrix[:, :, i], index=class_names_eval,
                             columns=class_names_eval)
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=0.8)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g')  # font size
        plt.savefig(figures_path + '/' + opts.dataset+'_top_' + str(i) + '.png')
        plt.figure().clear()


def main():
    device = init()
    opt = opts().parse()
    eval(opt, debug_data=opt.debug_data_mini, debug_visualize=opt.debug_visualize, device=device)


if __name__ == '__main__':
    main()

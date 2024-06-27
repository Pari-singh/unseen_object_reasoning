import os
import numpy as np
import pdb
import copy
import random
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

## PyTorch
import torch
import torch.utils.data as data
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import mlflow.pytorch
from mlflow.tracking import MlflowClient

import novel_objects.src.utils as utils
import novel_objects.src.network as network
from novel_objects.src.opts import opts as opts
from novel_objects.src.dataset.imagenet import ImageNetMini as data_source
mlflow.pytorch.autolog()

def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

def init():
    random.seed(42)
    matplotlib.rcParams['lines.linewidth'] = 2.0
    sns.reset_orig()
    sns.set()
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    return device


def train_model(latent_dim,
                use_pretrained=False,
                use_mode_loss=True,
                mode_loss_weights=None,
                num_classes=None,
                train_loader=None,
                val_loader=None,
                opts=None,
                device=None,
                train_dataset=None):

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(opts.ckpt_path, opts.dataset + f"_{latent_dim}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=opts.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    network.GenerateCallback(utils.get_train_images(8, train_dataset),
                                                             every_n_epochs=10),
                                    LearningRateMonitor("epoch")],
                         auto_lr_find=True,
                         log_every_n_steps=opts.log_every_n_steps)
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(opts.ckpt_path, opts.dataset + f"_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename) and use_pretrained:
        print("Found pretrained model, loading...")
        model = network.Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        if opts.network == 'default':
            model = network.Autoencoder(base_channel_size=32, latent_dim=latent_dim, use_mode_loss=use_mode_loss,
                                        mode_loss_weights=mode_loss_weights, width=opts.img_width, height=opts.img_height,
                                        num_classes=num_classes, opts=opts)
        elif opts.network == 'res50':
            model = network.Autoencoder(use_mode_loss=use_mode_loss,
                                        mode_loss_weights=mode_loss_weights,
                                        num_classes=num_classes, opts=opts)

        if opts.use_mlflow:
            with mlflow.start_run() as run:
                params = {
                    'dataset': opts.dataset,
                    'img_dims': opts.img_height,
                    'batch_size': opts.batch_size,
                    'latent_dims': opts.latent_dim,
                    'loss_w1': opts.loss_w1,
                    'loss_w2': opts.loss_w2,
                    'exponential_scale': opts.l2_scale,
                    'scheduler': opts.scheduler_type
                }
                mlflow.log_params(params)

                trainer.fit(model, train_loader, val_loader)
                print('callback metrics are:\n {}'.format(trainer.callback_metrics))

            # fetch the auto logged parameters and metrics
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    result = {"val": val_result}
    return model, result


def train(opts=None, device=None, visualize=True, debug_data=False):

    dataset_path = opts.dataset_path
    assert os.path.exists(dataset_path)

    # Path to the folder where the pretrained models are saved
    checkpoint_path = opts.ckpt_path
    project_home = opts.project_home

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(checkpoint_path, exist_ok=True)

    # class dataLoader
    # Transformations applied on each image => only make them a tensor
    use_augmentation = opts.use_augmentation

    if use_augmentation:
        transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=10),
                                        transforms.RandomGrayscale(p=0.1), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ColorJitter(brightness=.5, hue=.3),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = data_source(root=dataset_path, train=True, transform=transform,
                                download=True, debug_dataset_mini=opts.debug_data_mini,
                                use_patch=opts.use_patch_op, patch_res = opts.patch_resolution)

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
    test_dataset.data, test_dataset.targets, test_dataset.num_classes = \
        test_dataset.data[train_idx], [test_dataset.targets[x] for x in test_idx], len(test_classes)

    num_classes = trainval_dataset.num_classes
    train_set, val_set = torch.utils.data.random_split(trainval_dataset, [train_split, val_split])

    # create train & val loaders
    train_loader = data.DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=opts.batch_size, shuffle=False, drop_last=False, num_workers=4)
    # test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    if visualize:
        utils.visualization(train_dataset, trainval_dataset, train_split_map)

    # initial model dict
    model_dict = {}
    mode_loss_weights = [opts.loss_w1, opts.loss_w2]
    use_mode_loss = True
    latent_dim = opts.latent_dim

    # train
    if opts.training_lib == 'lightning':
        model_ld, result_ld = train_model(latent_dim, use_pretrained=False,
                                          use_mode_loss=use_mode_loss, mode_loss_weights=mode_loss_weights, opts=opts,
                                          num_classes=num_classes, train_loader=train_loader, val_loader=val_loader,
                                          train_dataset=trainval_dataset)
    else:
        model_ld, result_ld = [], []
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
    # save model
    torch.save(model_ld, os.path.join(opts.ckpt_path, opts.dataset + '_' + str(latent_dim)))


def main():
    device = init()
    opt = opts().parse()
    train(opt, device, visualize=opt.debug_visualize,
          debug_data=opt.debug_data_mini)


if __name__ == '__main__':
    main()

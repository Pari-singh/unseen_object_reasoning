import os
import numpy as np
import pdb
import copy
import random
import matplotlib
import pytorch_lightning.loggers
import seaborn as sns

## PyTorch
import torch
from torch.autograd import Variable
import torch.utils.data as data
from novel_objects.src.dataset.cifar10_seg import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import LightningModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything
import mlflow

import novel_objects.src.utils as utils
import novel_objects.src.network as network
from novel_objects.src.opts import opts as opts

mlflow.set_tracking_uri('http://127.0.0.1:5000')  # set up connection
mlflow.set_experiment('test-experiment2')          # set the experiment
mlflow.pytorch.autolog()


def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def train_model(latent_dim,
                use_pretrained=False,
                use_mode_loss=True,
                mode_loss_weights=None,
                num_classes=None,
                train_loader=None,
                val_loader=None,
                test_loader=None,
                opts=None,
                device=None,
                train_dataset=None):

    # setup logger
    trainer = pl.Trainer(default_root_dir=os.path.join(opts.ckpt_path, opts.dataset + f"_{latent_dim}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=opts.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True), MyPrintingCallback()],
                         auto_lr_find=True)
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(opts.ckpt_path, opts.dataset + f"_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename) and use_pretrained:
        print("Found pretrained model, loading...")
        model = network.Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        if opts.network == 'default':
            model = network.Autoencoder(base_channel_size=32, latent_dim=latent_dim,
                                        use_mode_loss=use_mode_loss,
                                        mode_loss_weights=mode_loss_weights, width=opts.img_width,
                                        height=opts.img_height,
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
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

def train(opts=None, device=None, visualize=True):

    dataset_path = opts.dataset_path
    assert os.path.exists(dataset_path)
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
    train_dataset = CIFAR10(root=dataset_path, train=True, transform=transform, download=True)
    pl.seed_everything(42)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
    # Loading the test set
    test_set = CIFAR10(root=dataset_path, train=False, transform=transform, download=True)

    # create train & val loaders
    train_loader = data.DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=opts.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    # initial model dict
    model_dict = {}
    mode_loss_weights = [opts.loss_w1, opts.loss_w2]
    use_mode_loss = True
    latent_dim = opts.latent_dim

    # train
    if opts.training_lib == 'lightning':
        model_ld, result_ld = train_model(latent_dim, use_pretrained=False,
                                          use_mode_loss=use_mode_loss, mode_loss_weights=mode_loss_weights, opts=opts,
                                          num_classes=10, train_loader=train_loader, val_loader=val_loader,
                                          train_dataset=train_dataset)
    else:
        model_ld, result_ld = [], []
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
    # save model
    torch.save(model_ld, os.path.join(opts.ckpt_path, opts.dataset + '_' + str(latent_dim)))

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


def main():
    device = init()
    opt = opts().parse()
    train(opt, device, visualize=opt.debug_visualize)


if __name__ == '__main__':
    main()

import os
import pdb

import numpy as np
import math
from torch.autograd import Variable

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# PyTorch Lightning
import pytorch_lightning as pl
from torch.optim import lr_scheduler

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 dataset: str,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        # self.c_hid = c_hid
        if dataset == 'imagenet_mini':
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32 # (32x32 => 16x16)
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16 (16x16 => 8x8)
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8 (8x8 => 4x4)
                act_fn(),
                nn.Conv2d(4 * c_hid, 4 * c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(4 * c_hid, 4 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
                act_fn(),
                nn.Flatten(),  # Image grid to single feature vector
                nn.Linear(4 * 16 * c_hid, latent_dim)
            )
        if dataset == 'CIFAR10':
            self.net = nn.Sequential(
                nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
                act_fn(),
                nn.Flatten(),  # Image grid to single feature vector
                nn.Linear(2 * 16 * c_hid, latent_dim)
            )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 dataset: str,
                 act_fn: object = nn.GELU,
                 num_classes: int = -1):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size

        if dataset == 'imagenet_mini':
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 4 * 16 * c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                nn.ConvTranspose2d(4 * c_hid, 4 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
                # 4x4 => 8x8
                act_fn(),
                nn.Conv2d(4 * c_hid, 4 * c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(4 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
                # 8x8 => 16x16
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                act_fn(),

                # 16x16 => 32x32
            )
            self.final = nn.Sequential(
                nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(c_hid, num_classes, kernel_size=3, output_padding=1, padding=1,
                                   stride=2),  # 32x32 => 64x64
                nn.Softmax()
                # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            )
            # if use_patch:
            #     self.final = nn.Sequential(
            #         nn.ConvTranspose2d(2 * c_hid, num_classes, kernel_size=3, output_padding=1, padding=1, stride=2),
            #         nn.Softmax()
            #     )
            # else:
            #     self.final = nn.Sequential(
            #         nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            #         act_fn(),
            #         nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            #         act_fn(),
            #         nn.ConvTranspose2d(c_hid, num_classes, kernel_size=3, output_padding=1, padding=1,
            #                            stride=2),  # 32x32 => 64x64
            #         nn.Softmax()
            #         # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            #     )
            self.net = nn.Sequential(self.net, self.final)

        if dataset == 'CIFAR10':
            self.linear = nn.Sequential(
                nn.Linear(latent_dim, 2 * 16 * c_hid),
                act_fn()
            )
            self.net = nn.Sequential(
                nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
                # 4x4 => 8x8
                act_fn(),
                nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
                act_fn(),
                )
            self.final = nn.Sequential(
                nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
                # 8x8 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
                act_fn(),
                nn.ConvTranspose2d(c_hid, num_classes, kernel_size=3, output_padding=1, padding=1,
                                   stride=2),  # 16x16 => 32x32
                nn.Softmax()
                # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            )
            # if use_patch:
            #     self.final = nn.Sequential(
            #         nn.ConvTranspose2d(2 * c_hid, num_classes, kernel_size=3, output_padding=1, padding=1, stride=2),
            #         nn.Softmax()
            #     )
            # else:
            #     self.final = nn.Sequential(
            #         nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            #         # 8x8 => 16x16
            #         act_fn(),
            #         nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            #         act_fn(),
            #         nn.ConvTranspose2d(c_hid, num_classes, kernel_size=3, output_padding=1, padding=1,
            #                            stride=2),  # 16x16 => 32x32
            #         nn.Softmax()
            #         # The input images is scaled between -1 and 1, hence the output has to be bounded as well
            #     )
            self.net = nn.Sequential(self.net, self.final)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 3,
                 width: int = -1,
                 height: int = -1,
                 num_classes: int = -1,
                 use_mode_loss: bool = True,
                 mode_loss_weights: list = None,
                 opts=None):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.opts = opts
        dataset = self.opts.dataset
        use_patch = self.opts.use_patch_op
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim, dataset)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim, dataset,
                                     num_classes=num_classes)
        # Example input array needed for visualizing the graph of the network
        # self.example_input_array = torch.zeros(6, num_input_channels, width, height)
        self._num_classes = num_classes
        self._use_mode_loss = use_mode_loss
        self.criterion = nn.CrossEntropyLoss()
        self.mode_loss_weights = mode_loss_weights
        if mode_loss_weights is None:
            self.mode_loss_weights = [opts.mode_loss_ratio, 1.0 - opts.mode_loss_ratio]

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # pdb.set_trace()
        return x_hat

    def _get_entropy_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, classes, classes_ohe, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        if self._use_mode_loss:
            loss_mode = self.mode_loss(classes, x_hat)
            loss_reg = self.regressionl1loss(classes_ohe, x_hat)
            loss = self.mode_loss_weights[0] * loss_mode + self.mode_loss_weights[1] * loss_reg
        else:
            loss = self.pixel_class_loss(classes, x_hat)
        return loss

    @staticmethod
    def one_cycle(y1=0.0, y2=1.0, steps=100):
        # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
        return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.opts.lr)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        if self.opts.scheduler_type in ['reduce_lr_on_plateau', 'stepLR', 'multistepLR']:
            if self.opts.scheduler_type == 'reduce_lr_on_plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                 mode='min',
                                                                 factor=self.opts.factor,
                                                                 patience=self.opts.patience,
                                                                 min_lr=self.opts.min_lr)
            elif self.opts.scheduler_type == 'stepLR':
                print("Using Step LR")
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opts.patience, gamma=self.opts.factor)

            elif self.opts.scheduler_type == 'multistepLR':

                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150, 350, 600, 900],
                                                           gamma=self.opts.factor)
            else:
                print("Scheduler Error: Not defined")

        elif self.opts.scheduler_type in ['linear', 'one_cycle']:
            # Scheduler
            if self.opts.scheduler_type == 'linear':
                lf = lambda x: (1 - x / (self.opts.max_epochs - 1)) * (1.0 - self.opts.lrf) + self.opts.lrf  # linear
            elif self.opts.scheduler_type == 'one_cycle':
                lf = self.one_cycle(1, self.opts.lrf, self.opts.max_epochs)  # cosine 1->hyp['lrf']
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        else:
            scheduler = None
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_entropy_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_entropy_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_entropy_loss(batch)
        self.log('test_loss', loss)

    def pixel_class_loss(self, y, x_hat):
        loss = torch.nn.CrossEntropyLoss()
        output = loss(x_hat, y.long())
        return output

    def regressionl1loss(self, y, x_hat):
        loss = torch.nn.L1Loss()
        output = loss(x_hat, y)
        return output

    def mode_loss(self, y_labels, x_hat):
        x_labels = torch.argmax(x_hat, dim=1)
        #TODO : create patch of 3x3 instead of flatten
        y_mode, _ = torch.mode(torch.flatten(y_labels, start_dim=1))
        x_mode, _ = torch.mode(torch.flatten(x_labels, start_dim=1))
        # x_mean = torch.mean(torch.flatten(x_hat, start_dim=2), dim=2)
        x_mode_ohe = nn.functional.one_hot(x_mode, self._num_classes)
        y_mode_ohe = nn.functional.one_hot(y_mode.long(), self._num_classes)
        if self.opts.use_l2:
            loss = torch.nn.MSELoss()
            output = loss(self.opts.l2_scale * torch.exp(x_mode_ohe.float()),
                          self.opts.l2_scale * torch.exp(y_mode_ohe.float()))
        else:
            loss = torch.nn.L1Loss()
            output = loss(x_mode_ohe.float(), y_mode_ohe.float())
        return output


class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
    # def on_save_checkpoint(
    #     self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    # ) -> dict:
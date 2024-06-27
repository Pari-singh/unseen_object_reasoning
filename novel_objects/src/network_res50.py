import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import math

import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 auto_encoder_class: object = UNetWithResnet50Encoder,
                 num_classes: int = -1,
                 use_mode_loss: bool = True,
                 mode_loss_weights: list = None,
                 opts=None):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder_decoder = auto_encoder_class(n_classes=num_classes)
        self._num_classes = num_classes
        self._use_mode_loss = use_mode_loss
        self.criterion = nn.CrossEntropyLoss()
        self.mode_loss_weights = mode_loss_weights
        self.opts = opts
        if mode_loss_weights is None:
            self.mode_loss_weights = [opts.mode_loss_ratio, 1.0 - opts.mode_loss_ratio]

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        x_hat = self.encoder_decoder(x)
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
        # pdb.set_trace()
        output = loss(x_hat, y)
        return output

    def mode_loss(self, y_labels, x_hat):
        x_labels = torch.argmax(x_hat, dim=1)
        y_mode, _ = torch.mode(torch.flatten(y_labels, start_dim=1))
        x_mode, _ = torch.mode(torch.flatten(x_labels, start_dim=1))
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


import os
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from torch.autograd import Variable

import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()
import pdb
import pandas as pd
import seaborn as sn

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

# Torchvision
import torchvision
from novel_objects.src.dataset.cifar10_seg import CIFAR10
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import urllib.request
from urllib.error import HTTPError

# Tensorboard extension (for visualization purposes later)
from torch.utils.tensorboard import SummaryWriter

# ### Global params
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/home/auro/paridhi/Experimental/saved_models"
PROJECT_HOME = "/home/auro/paridhi/Experimental/novel_objects"

# Setting the seed
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# class dataLoader
# Transformations applied on each image => only make them a tensor
use_augmentation = True

if use_augmentation:
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(degrees=10),
                                    transforms.RandomGrayscale(p=0.1), transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=.5, hue=.3),
                                    transforms.Normalize((0.5,), (0.5,))])
else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

# Loading the training dataset. We need to split it into a training and validation part
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform,
                        download=True, use_patch=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

# Loading the test set
test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform,
                   download=True, use_patch=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)


def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
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
                 act_fn: object = nn.GELU,
                 num_classes: int = 10):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
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
            # nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # # 8x8 => 16x16
            # act_fn(),
            # nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            # act_fn(),
            nn.ConvTranspose2d(2* c_hid, num_classes, kernel_size=3, output_padding=1, padding=1,
                               stride=2),  # 16x16 => 32x32
            nn.Softmax()
            # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

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
                 width: int = 32,
                 height: int = 32,
                 num_classes: int = 10,
                 use_mode_loss: bool = True,
                 mode_loss_weights: list = None):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(6, num_input_channels, width, height)
        self._num_classes = num_classes
        self._use_mode_loss = use_mode_loss
        self.criterion = nn.CrossEntropyLoss()
        self.mode_loss_weights = mode_loss_weights
        if mode_loss_weights is None:
            self.mode_loss_weights = [0.7, 0.3]

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_entropy_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, classes, classes_ohe, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        if self._use_mode_loss:
            loss_mode = self.mode_loss(classes, x_hat)
            # loss_reg = self.regressionl1loss(classes_ohe, x_hat)
            loss = self.mode_loss_weights[0] * loss_mode + self.mode_loss_weights[1] * loss_reg
        else:
            loss = self.pixel_class_loss(classes, x_hat)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
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
        y_mode, _ = torch.mode(torch.flatten(y_labels, start_dim=1))
        x_mode, _ = torch.mode(torch.flatten(x_labels, start_dim=1))
        # x_mean = torch.mean(torch.flatten(x_hat, start_dim=2), dim=2)
        x_mode_ohe = nn.functional.one_hot(x_mode, self._num_classes)
        y_mode_ohe = nn.functional.one_hot(y_mode.long(), self._num_classes)

        loss = torch.nn.L1Loss()
        output = loss(x_mode_ohe.float(), y_mode_ohe.float())

        # loss_l2 = torch.nn.MSELoss()
        # output = loss_l2(4.0*torch.exp(x_mode_ohe.float()), 4.0*torch.exp(y_mode_ohe.float()))
#         pdb.set_trace()
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


def train_cifar(latent_dim, use_pretrained=False, use_mode_loss=True, mode_loss_weights=None):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=1000,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(get_train_images(8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")],
                         auto_lr_find=True)
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"CIFAR10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename) and use_pretrained:
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim, use_mode_loss=use_mode_loss,
                            mode_loss_weights=mode_loss_weights)
        # lr_finder = trainer.tuner.lr_find(model)
        # model.learning_rate = lr_finder.suggestion()
        # trainer.tune(model)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


############################# MAIN CODING ######################################
model_dict = {}
mode_loss_weights = [10, 5]
use_mode_loss = True
eval_only = True
if not eval_only:
    for latent_dim in [384]:
        model_ld, result_ld = train_cifar(latent_dim, use_pretrained=False,
                                          use_mode_loss=use_mode_loss, mode_loss_weights=mode_loss_weights)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
        torch.save(model_ld.state_dict(), os.path.join(CHECKPOINT_PATH, 'CIFAR10_'+str(latent_dim)))


# test on validation dat
def eval_cifar(latent_dim):
    evaluator = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}"),
                           gpus=1 if str(device).startswith("cuda") else 0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        raise ValueError("Pretrained weights not found")

    # Test best model on validation and test set
    val_result = evaluator.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = evaluator.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

# ## INFERENCE
#############################
# run on sample data

exp_id = 2
latent_dim = 384
ckpt_file_name = 'epoch=78-step=27728.ckpt'
class_names_cifar_10 = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
class_names_cifar_10_rearranged = ['airplane', 'automobile', 'ship', 'truck',
                                   'bird', 'cat', 'deer',
                                   'dog', 'frog', 'horse']
ckpt_path = 'saved_models/CIFAR10_' + str(latent_dim) + '/lightning_logs/version_' + str(exp_id) + '/checkpoints'
ckpt = os.path.join(PROJECT_HOME, ckpt_path, ckpt_file_name)

model = Autoencoder.load_from_checkpoint(ckpt)

confusion_matrix = np.zeros((len(class_names_cifar_10), len(class_names_cifar_10), 3), dtype=np.int32)
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
    print(i,  ' /', len(val_loader.dataset.indices) )

figures_path = os.path.join(PROJECT_HOME, ckpt_path.rsplit('/', 1)[0], 'figures')
os.makedirs(figures_path, exist_ok=True)
for i in range(3):
    pd.set_option('display.float_format', '{:.2f}'.format)
    df_cm = pd.DataFrame(confusion_matrix[:, :, i], index=class_names_cifar_10,
                         columns=class_names_cifar_10)
    # df_cm.reindex(class_names_cifar_10_rearranged)
    plt.figure(figsize=(10,7))
    sn.set(font_scale=0.8)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g')  # font size
    # plt.show()
    plt.savefig(figures_path + '/' + 'cifar10_top_' + str(i) + '.png')
    plt.figure().clear()

imgplot = plt.imshow(raw_img)
plt.show()
print('Objects class: ' + class_names_cifar_10[sample_class])
print({class_names_cifar_10[i]: weights[i] for i in range(len(weights))})
sorted_idx = np.flip(np.argsort(weights))

sample_class_loc = np.where(sorted_idx==sample_class)[0][0]
if sample_class_loc == 0:
    sorted_idx = sorted_idx[1:]
elif sample_class_loc == len(sorted_idx)-1:
    sorted_idx = sorted_idx[:-1]
else:
    sorted_idx = np.concatenate((sorted_idx[:sample_class_loc],
                                 sorted_idx[(sample_class_loc+1):]))

print("Most similar classes: ", {class_names_cifar_10[x]: weights[x] for x in sorted_idx[:3]})
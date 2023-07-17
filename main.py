import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
from training_utils import train, test
from torchvision import datasets
import numpy as np


# %%
# Setup paths for data loading and results.
# ---------------------------------------------------------------

dataset_name = "fashionMNIST"  # options are MNIST, fashionMNIST, CelebA, flowers
undersampling_ratio = .3  # ratio of measurements to original image size
number_of_operators = 10  # number of different random operators


# %%
# Setup paths for data loading and results.
# ---------------------------------------------------------------

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------
# In this example, we use the MNIST dataset for training and testing.
#


if dataset_name == "MNIST":
    transform = transforms.Compose([transforms.ToTensor()])
    train_base_dataset = datasets.MNIST(
        root="../datasets/", train=True, transform=transform, download=True
    )
    test_base_dataset = datasets.MNIST(
        root="../datasets/", train=False, transform=transform, download=True
    )
elif dataset_name == "fashionMNIST":
    transform = transforms.Compose([transforms.ToTensor()])
    train_base_dataset = datasets.FashionMNIST(
        root="../datasets/", train=True, transform=transform, download=True
    )
    test_base_dataset = datasets.FashionMNIST(
        root="../datasets/", train=False, transform=transform, download=True
    )
elif dataset_name == "CelebA":
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(128)])
    train_base_dataset = datasets.CelebA(
        root="../datasets/", split="train", transform=transform, download=True
    )
    test_base_dataset = datasets.CelebA(
        root="../datasets/", split="test", transform=transform, download=True
    )
elif dataset_name == "flowers":
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(128)])
    train_base_dataset = datasets.Flowers102(
        root="../datasets/", split="train", transform=transform, download=True
    )
    test_base_dataset = datasets.Flowers102(
        root="../datasets/", split="test", transform=transform, download=True
    )
else:
    raise ValueError("Dataset name not found")

# %%
# Generate a dataset of subsampled images and load it.
# ----------------------------------------------------------------------------------
# We generate 10 different inpainting operators, each one with a different random mask.
# If the :func:`deepinv.datasets.generate_dataset` receives a list of physics operators, it
# generates a dataset for each operator and returns a list of paths to the generated datasets.
#
# .. note::
#
#   We only use 10 training images per operator to reduce the computational time of this example. You can use the whole
#   dataset by setting ``n_images_max = None``.

n = np.prod(train_base_dataset[0][0].shape)
channels = train_base_dataset[0][0].shape[0]
# defined physics
physics = [
    dinv.physics.CompressedSensing(m=int(undersampling_ratio*n), img_shape=train_base_dataset[0][0].shape,
                                   sensor_model=lambda x: torch.sign(x), device=device)
    for _ in range(number_of_operators)
]

# Use parallel dataloader if using a GPU to reduce training time,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0
n_images_max = (
    None if torch.cuda.is_available() else 50
)  # number of images used for training (uses the whole dataset if you have a gpu)

operation = "onebit_cs"
my_dataset_name = "demo_ssbm"
measurement_dir = DATA_DIR / "MNIST" / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    test_datapoints=10,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = [
    dinv.datasets.HDF5Dataset(path=path, train=True) for path in deepinv_datasets_path
]
test_dataset = [
    dinv.datasets.HDF5Dataset(path=path, train=False) for path in deepinv_datasets_path
]

# %%
# Set up the reconstruction network
# ---------------------------------------------------------------
#
# As a reconstruction network, we use a simple artifact removal network based on a U-Net.
# The network is defined as a :math:`R_{\theta}(y,A)=\phi_{\theta}(A^{\top}y)` where :math:`\phi` is the U-Net.

# Define the unfolded trainable model.
model = dinv.models.ArtifactRemoval(
    backbone_net=dinv.models.UNet(in_channels=channels, out_channels=channels, scales=3)
)
model = model.to(device)

# %%
# Set up the training parameters
# --------------------------------------------
# We choose a self-supervised training scheme with two losses: the measurement consistency loss (MC)
# and the multi-operator imaging loss (MOI).
#
# .. note::
#
#       We use a pretrained model to reduce training time. You can get the same results by training from scratch
#       for 100 epochs.

epochs = 100
learning_rate = 5e-4
batch_size = 128 if torch.cuda.is_available() else 1

# choose self-supervised training losses
# generates 4 random rotations per image in the batch

if number_of_operators == 1:
    losses = [dinv.loss.MCLoss(), dinv.loss.EILoss(dinv.transform.Shift(4))]
else:
    losses = [dinv.loss.MCLoss(), dinv.loss.MOILoss(physics)]

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)


# %%
# Train the network
# --------------------------------------------
#
#

verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

train_dataloader = [
    DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    for dataset in train_dataset
]
test_dataloader = [
    DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for dataset in test_dataset
]

train(
    model=model,
    train_dataloader=train_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=physics,
    optimizer=optimizer,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
    ckp_interval=20,
)

# %%
# Test the network
# --------------------------------------------
#
#

plot_images = True

test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=plot_images,
    save_folder=RESULTS_DIR / "ssbm" / operation,
    verbose=verbose,
    wandb_vis=wandb_vis,
)

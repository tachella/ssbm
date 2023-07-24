import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
from training_utils import test, train
from torchvision import datasets
from biht import BIHT
import numpy as np

# Setup learning problem
# --------------------------------------------

dataset_name = "fashionMNIST"  # options are "MNIST", "fashionMNIST", "CelebA", "flowers"
undersampling_ratio = .385  # ratio of measurements to original image size
number_of_operators = 10  # number of different random operators.
method = 'ssbm_multop' # options are "ssbm_multop" (proposed multioperator loss),
# "ssbm_equiv" (proposed equivariant loss), "measurement_consistency", "supervised" and
# "biht" (binary iterative hard thresholding with wavelet basis).


# Setup paths for data loading and results.
# --------------------------------------------
BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"

verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Load base image dataset.
# --------------------------------------------

if dataset_name == "MNIST":
    transform = transforms.Compose([transforms.ToTensor()])
    train_base_dataset = datasets.MNIST(
        root="../datasets/", train=True, transform=transform, download=True,
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


# Generate a dataset of binary measurements from the base dataset.
# --------------------------------------------

n = np.prod(train_base_dataset[0][0].shape)
channels = train_base_dataset[0][0].shape[0]
# defined physics operators
physics = [
    dinv.physics.CompressedSensing(m=int(undersampling_ratio*n), img_shape=train_base_dataset[0][0].shape,
                                   sensor_model=lambda x: torch.sign(x), device=device)
    for _ in range(number_of_operators)
]

# Use parallel dataloader if using a GPU to reduce training time,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0

operation = "onebit_cs"
my_dataset_name = "demo_ssbm"
measurement_dir = DATA_DIR / dataset_name / operation

deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

if not isinstance(deepinv_datasets_path, list):
    deepinv_datasets_path = [deepinv_datasets_path]

train_dataset = [
    dinv.datasets.HDF5Dataset(path=path, train=True) for path in deepinv_datasets_path
]
test_dataset = [
    dinv.datasets.HDF5Dataset(path=path, train=False) for path in deepinv_datasets_path
]


# Set up the reconstruction method
# --------------------------------------------
batch_size = 128 if torch.cuda.is_available() else 1

if method == 'biht':
    max_iter = 100
    coeffs = 20
    stepsize = 10/np.sqrt(undersampling_ratio*n)
    model = BIHT(stepsize=stepsize,
                 max_iter=max_iter, thresholded_coeffs=coeffs, device=device)
else:

    # As a reconstruction network, we use a simple artifact removal network based on a U-Net.
    # The network is defined as a :math:`R_{\theta}(y,A)=\phi_{\theta}(A^{\top}y)` where :math:`\phi` is the U-Net.

    # Define the unfolded trainable model.
    model = dinv.models.ArtifactRemoval(
        backbone_net=dinv.models.UNet(in_channels=channels, out_channels=channels, scales=3)
    )
    model = model.to(device)

    # Set up the training losses
    # --------------------------------------------
    if method == 'ssbm_multop':
        losses = [dinv.loss.MCLoss(metric=torch.nn.SoftMarginLoss()), dinv.loss.MOILoss(physics, weight=.1)]
    elif method == 'ssbm_equiv':
        losses = [dinv.loss.MCLoss(metric=torch.nn.SoftMarginLoss()),
                  dinv.loss.EILoss(dinv.transform.Shift(4), weight=.1)]
    elif method == 'supervised':
        losses = [dinv.loss.SupLoss()]
    elif method == 'measurement_consistency':
        losses = [dinv.loss.MCLoss(metric=torch.nn.SoftMarginLoss())]
    else:
        raise ValueError("Training loss not found")

    epochs = 100
    learning_rate = 1e-4

    # choose optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)

    # Train the network
    train_dataloader = [
        DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        for dataset in train_dataset
    ]
    test_dataloader = [
        DataLoader(torch.utils.data.Subset(dataset, torch.arange(50)),
                   batch_size=batch_size, num_workers=num_workers, shuffle=False)
        for dataset in test_dataset
    ]

    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
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

# Test the reconstruction algorithm
# --------------------------------------------

test_dataloader = [
    DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    for dataset in test_dataset
]

test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=True,
    save_folder=RESULTS_DIR / method / operation,
    verbose=verbose,
    wandb_vis=wandb_vis,
)

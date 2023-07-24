import deepinv as dinv


class OneSidedL1(dinv.optim.DataFidelity):
    def __init__(self):
        super().__init__()

    def grad(self, x, y, physics):
        return physics.A_adjoint(physics(x)-y)


def BIHT(stepsize, max_iter=100, thresholded_coeffs=20, device='cpu'):
    # Select the data fidelity term
    data_fidelity = OneSidedL1()

    prior = dinv.optim.PnP(dinv.models.WaveletPrior(non_linearity='topk').to(device))
    params_algo = {"stepsize": stepsize, "lambda": 1.0, "g_param": thresholded_coeffs}

    # instantiate the algorithm class to solve the IP problem.
    model = dinv.optim.optim_builder(
        iteration="PGD",
        prior=prior,
        data_fidelity=data_fidelity,
        verbose=True,
        max_iter=max_iter,
        params_algo=params_algo,
    )

    return model


# this code tests the BIHT algorithm on a single image. If you want to test it on the whole test dataset,
# please use main.py with the option method="biht"
if __name__ == "__main__":
    import torch
    from torchvision import datasets, transforms
    from deepinv.utils.metric import norm
    import numpy as np

    dataset_name = "fashionMNIST"
    undersampling_ratio = .5
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        test_base_dataset = datasets.MNIST(
            root="../datasets/", train=False, transform=transform, download=True
        )
    elif dataset_name == "fashionMNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        test_base_dataset = datasets.FashionMNIST(
            root="../datasets/", train=False, transform=transform, download=True
        )
    elif dataset_name == "CelebA":
        transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(128)])
        test_base_dataset = datasets.CelebA(
            root="../datasets/", split="test", transform=transform, download=True
        )
    elif dataset_name == "flowers":
        transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(128)])
        test_base_dataset = datasets.Flowers102(
            root="../datasets/", split="test", transform=transform, download=True
        )
    else:
        raise ValueError("Dataset name not found")

    # load the image
    img = test_base_dataset[0][0].unsqueeze(0)
    n = np.prod(test_base_dataset[0][0].shape)

    physics = dinv.physics.CompressedSensing(m=int(undersampling_ratio*n), img_shape=test_base_dataset[0][0].shape,
                                   sensor_model=lambda x: torch.sign(x), device=device)

    x = test_base_dataset[0][0].unsqueeze(0).to(device)
    y = physics(x)

    # instantiate the algorithm class to solve the IP problem.
    model = BIHT(stepsize=10/np.sqrt(undersampling_ratio*n),
                 max_iter=100, thresholded_coeffs=20, device=device)

    # run reconstruction
    x_hat, metrics = model(y, physics, compute_metrics=True)

    x_hat *= norm(x)/norm(x_hat)

    print(f'PSNR: {dinv.utils.cal_psnr(x, x_hat):.2f} dB')
    dinv.utils.plot([x, x_hat])

    dinv.utils.plot_curves(metrics, show=True)

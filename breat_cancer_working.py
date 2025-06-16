#!/usr/bin/env python3
"""
MIWAE-EBM: Missing data imputation using Variational Autoencoders and Energy-Based Models
"""

# Installation and imports
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.datasets import load_iris, load_breast_cancer
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


# Configuration
class Config:
    SEED = 1234
    HIDDEN_DIM = 128
    LATENT_DIM = 1
    K_SAMPLES = 20
    BATCH_SIZE = 64
    N_EPOCHS = 2002
    LEARNING_RATE = 1e-3
    MISSING_RATIO = 0.5
    MNAR_SIZE_RATIO = 0.5
    MCAR_SIZE_RATIO = 0.25
    DATASET = "breast_cancer"  # Options: "breast_cancer", "mnist", "custom"


# Utility functions
def mse(xhat, xtrue, mask):
    """Calculate MSE for imputations"""
    xhat = np.array(xhat)
    xtrue = np.array(xtrue)
    return np.mean(np.power(xhat - xtrue, 2)[~mask])


def setup_device():
    """Setup and print device information"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def weights_init(layer):
    """Initialize weights for neural network layers"""
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight)


# Data loading and preprocessing
def load_data():
    """Load and preprocess data based on the selected dataset"""
    if Config.DATASET == "breast_cancer":
        # Load breast cancer dataset
        data = load_breast_cancer()["data"]

        # Normalize data
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        n, p = data.shape
        data = np.random.permutation(data)  # Shuffle the data

    elif Config.DATASET == "mnist":
        # Load MNIST dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        )
        mnist = datasets.MNIST(root=".", train=True, download=True, transform=transform)

        data = mnist.data.float().view(-1, 784).numpy() / 255.0
        n, p = data.shape

    elif Config.DATASET == "custom":
        # Load custom dataset
        data = pd.read_csv(Config.CUSTOM_DATA_PATH).values

    else:
        raise ValueError(f"Unknown dataset: {Config.DATASET}")

    return data, n, p


def create_mnar_mask(data):
    """Create MAR mask for MNIST dataset"""
    masks = np.zeros((data.shape[0], data.shape[1]))
    print(data.shape)
    for i, example in enumerate(data):
        h = (1.0 / (784.0 / 2.0)) * np.sum(example[:392]) + 0.3
        pi = np.random.binomial(2, h)
        _mask = np.ones(example.shape[0])
        if pi == 0:
            _mask[196:392] = 0
        elif pi == 1:
            _mask[:392] = 0
        elif pi == 2:
            _mask[:196] = 0
        masks[i, :] = _mask
    return masks


def create_missing_data(data, n):
    """Create MNAR, MCAR, or MAR missing data patterns based on the dataset"""
    if Config.DATASET == "mnist":
        # Create MAR mask for MNIST
        mask_mnar = create_mnar_mask(data)
        data_nan_mnar = np.copy(data)
        data_nan_mnar[mask_mnar == 0] = np.nan

        # Create MCAR mask for MNIST
        np.random.seed(Config.SEED)
        mask_mcar = np.random.rand(*data.shape) < Config.MISSING_RATIO
        data_nan_mcar = np.copy(data)
        data_nan_mcar[mask_mcar] = np.nan

        return {
            "mnar": (data_nan_mnar, mask_mnar.astype(bool), data),
            "mcar": (data_nan_mcar, mask_mcar.astype(bool), data),
        }

    else:
        # Default MNAR and MCAR patterns for tabular datasets
        mnar_size = int(Config.MNAR_SIZE_RATIO * n)
        mcar_size = int(Config.MCAR_SIZE_RATIO * n)

        data_mnar = data[:mnar_size]
        data_mcar = data[mnar_size : mnar_size + mcar_size]
        data_test = data[mnar_size + mcar_size :]

        # Create MCAR pattern
        np.random.seed(Config.SEED)
        n_mcar, p = data_mcar.shape
        data_mcar_nan = np.copy(data_mcar)
        mask_mcar = np.random.rand(n_mcar) < Config.MISSING_RATIO
        mask_mcar = np.concatenate(
            [
                mask_mcar.reshape(-1, 1),
                np.ones((data_mcar.shape[0], data.shape[1] - 1)),
            ],
            axis=1,
        )

        for i in range(data_mcar.shape[0]):
            if not mask_mcar[i, 0]:
                data_mcar_nan[i, 0] = np.nan

        # Create MNAR pattern
        np.random.seed(Config.SEED)
        full_mean, full_std = np.mean(data, 0), np.std(data, 0)
        missing_mnar = data_mnar[:, 0] > full_mean[0] + 0.01 * full_std[0]
        data_mnar_nan = np.copy(data_mnar)
        mask = np.ones_like(data_mnar, dtype=bool)

        for i in range(data_mnar.shape[0]):
            if missing_mnar[i]:
                mask[i, 0] = False
                data_mnar_nan[i, 0] = np.nan

        return {
            "mnar": (data_mnar_nan, mask, data_mnar),
            "mcar": (data_mcar_nan, mask_mcar.astype(bool), data_mcar),
        }


# Neural network models
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * output_dim),
        )

    def forward(self, x):
        return self.net(x)


class EBM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64x7x7
            nn.ReLU(),
            nn.Flatten(),  # Flatten to 64*7*7
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Reshape to (batch_size, 1, 28, 28)
        features = self.conv_net(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return torch.cat([mu, logvar], dim=1)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 1x28x28
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)  # Reshape to (batch_size, 64, 7, 7)
        x = self.deconv_net(x)
        return x.view(-1, 28 * 28)  # Flatten to match the original MNIST shape


# MIWAE functions
def miwae_loss(iota_x, mask, encoder, decoder, p_z, K, d, p, device):
    """Calculate MIWAE loss"""
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(
        td.Normal(
            loc=out_encoder[..., :d],
            scale=torch.nn.Softplus()(out_encoder[..., d : (2 * d)]),
        ),
        1,
    )

    zgivenx = q_zgivenxobs.rsample([K])
    zgivenx_flat = zgivenx.reshape([K * batch_size, d])

    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p : (2 * p)]) + 0.001
    all_degfreedom_obs_model = (
        torch.nn.Softplus()(out_decoder[..., (2 * p) : (3 * p)]) + 3
    )

    data_flat = torch.Tensor.repeat(iota_x, [K, 1]).reshape([-1, 1])
    tiledmask = torch.Tensor.repeat(mask, [K, 1])

    all_log_pxgivenz_flat = torch.distributions.StudentT(
        loc=all_means_obs_model.reshape([-1, 1]),
        scale=all_scales_obs_model.reshape([-1, 1]),
        df=all_degfreedom_obs_model.reshape([-1, 1]),
    ).log_prob(data_flat)

    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([K * batch_size, p])
    logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([K, batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))
    return neg_bound


def miwae_impute(iota_x, mask, encoder, decoder, p_z, L, d, p, device):
    """Perform MIWAE imputation"""
    batch_size = iota_x.shape[0]
    out_encoder = encoder(iota_x)
    q_zgivenxobs = td.Independent(
        td.Normal(
            loc=out_encoder[..., :d],
            scale=torch.nn.Softplus()(out_encoder[..., d : (2 * d)]),
        ),
        1,
    )

    zgivenx = q_zgivenxobs.rsample([L])
    zgivenx_flat = zgivenx.reshape([L * batch_size, d])

    out_decoder = decoder(zgivenx_flat)
    all_means_obs_model = out_decoder[..., :p]
    all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., p : (2 * p)]) + 0.001
    all_degfreedom_obs_model = (
        torch.nn.Softplus()(out_decoder[..., (2 * p) : (3 * p)]) + 3
    )

    data_flat = torch.Tensor.repeat(iota_x, [L, 1]).reshape([-1, 1]).to(device)
    tiledmask = torch.Tensor.repeat(mask, [L, 1]).to(device)

    all_log_pxgivenz_flat = torch.distributions.StudentT(
        loc=all_means_obs_model.reshape([-1, 1]),
        scale=all_scales_obs_model.reshape([-1, 1]),
        df=all_degfreedom_obs_model.reshape([-1, 1]),
    ).log_prob(data_flat)

    all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])
    logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([L, batch_size])
    logpz = p_z.log_prob(zgivenx)
    logq = q_zgivenxobs.log_prob(zgivenx)

    xgivenz = td.Independent(
        td.StudentT(
            loc=all_means_obs_model,
            scale=all_scales_obs_model,
            df=all_degfreedom_obs_model,
        ),
        1,
    )

    imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq, 0)
    xms = xgivenz.sample().reshape([L, batch_size, p])
    xm = torch.einsum("ki,kij->ij", imp_weights, xms)

    return xm


def train_baselines(data_mnar_nan, data_mnar, mask):
    """Train baseline imputation methods"""
    # MissForest
    missforest = IterativeImputer(
        max_iter=20, estimator=ExtraTreesRegressor(n_estimators=100)
    )
    missforest.fit(data_mnar_nan)
    xhat_mf = missforest.transform(data_mnar_nan)

    # Iterative Ridge
    iterative_ridge = IterativeImputer(max_iter=20, estimator=BayesianRidge())
    iterative_ridge.fit(data_mnar_nan)
    xhat_ridge = iterative_ridge.transform(data_mnar_nan)

    # Mean imputation
    mean_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    mean_imp.fit(data_mnar_nan)
    xhat_mean = mean_imp.transform(data_mnar_nan)

    return {
        "missforest": (xhat_mf, mse(xhat_mf, data_mnar, mask)),
        "ridge": (xhat_ridge, mse(xhat_ridge, data_mnar, mask)),
        "mean": (xhat_mean, mse(xhat_mean, data_mnar, mask)),
    }


def plot_results(mse_train, baselines):
    """Plot training results and baseline comparisons"""
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(mse_train) * 100 + 1, 100), mse_train, color="blue", label="MIWAE"
    )
    plt.axhline(
        y=baselines["missforest"][1], linestyle="-", color="red", label="missForest"
    )
    plt.axhline(
        y=baselines["ridge"][1], linestyle="-", color="orange", label="Iterative ridge"
    )
    plt.axhline(
        y=baselines["mean"][1], linestyle="-", color="green", label="Mean imputation"
    )
    plt.legend()
    plt.title("Imputation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()


import matplotlib.pyplot as plt


def plot_mnist_data(original_data, masked_data, mask, filename_prefix):
    """Plot and save MNIST data before and after masking."""
    # Reshape the data to 28x28 images
    original_images = original_data.reshape(-1, 28, 28)
    masked_images = masked_data.reshape(-1, 28, 28)
    mask_images = mask.reshape(-1, 28, 28)

    # Plot a few examples
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        # Original data
        axes[0, i].imshow(original_images[i], cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # Masked data
        axes[1, i].imshow(masked_images[i], cmap="gray")
        axes[1, i].set_title("Masked")
        axes[1, i].axis("off")

        # Mask
        axes[2, i].imshow(mask_images[i], cmap="gray")
        axes[2, i].set_title("Mask")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_mnist_data.png")
    plt.close()


def main():
    """Main training loop"""
    # Setup
    device = setup_device()
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)

    # Load and prepare data
    data, n, p = load_data()
    missing_data = create_missing_data(data, n)
    data_nan_mnar, mask_mnar, original_data_mnar = missing_data["mnar"]
    data_nan_mcar, mask_mcar, original_data_mcar = missing_data["mcar"]

    # If MNIST, plot and save pre-masking and masked data
    if Config.DATASET == "mnist":
        print("Saving MNIST data plots for sanity check...")
        plot_mnist_data(original_data_mnar, data_nan_mnar, mask_mnar, "mnar")
        plot_mnist_data(original_data_mcar, data_nan_mcar, mask_mcar, "mcar")

    # Initialize data for imputation
    xhat_0 = np.copy(data_nan_mnar)
    xhat_0[np.isnan(data_nan_mnar)] = 0

    # Initialize models
    if Config.DATASET == "mnist":
        encoder = ConvEncoder(Config.LATENT_DIM).to(device)
        decoder = ConvDecoder(Config.LATENT_DIM).to(device)
    else:
        encoder = Encoder(p, Config.HIDDEN_DIM, Config.LATENT_DIM).to(device)
        decoder = Decoder(Config.LATENT_DIM, Config.HIDDEN_DIM, p).to(device)
    d = Config.LATENT_DIM
    p_z = td.Independent(
        td.Normal(
            loc=torch.zeros(d, device=device), scale=torch.ones(d, device=device)
        ),
        1,
    )

    ebm = EBM(input_dim=p, hidden_dim=Config.HIDDEN_DIM).to(device)

    # Initialize weights
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    ebm.apply(weights_init)

    # Optimizers
    miwae_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=Config.LEARNING_RATE
    )
    ebm_optimizer = optim.Adam(ebm.parameters(), lr=Config.LEARNING_RATE)

    # Prepare data loaders
    xhat_0_tensor = torch.from_numpy(xhat_0).float()
    mask_tensor = torch.from_numpy(mask_mnar).float()
    dataset = TensorDataset(xhat_0_tensor, mask_tensor)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Training tracking
    mse_train = []
    xhat = np.copy(xhat_0)

    print("Starting MIWAE training...")

    # Phase 1: Train MIWAE
    for ep in range(1, Config.N_EPOCHS):
        for b_data, b_mask in loader:
            b_data = b_data.to(device)
            b_mask = b_mask.to(device)

            miwae_optimizer.zero_grad()
            loss = miwae_loss(
                b_data,
                b_mask,
                encoder,
                decoder,
                p_z,
                Config.K_SAMPLES,
                Config.LATENT_DIM,
                p,
                device,
            )
            loss.backward()
            miwae_optimizer.step()

        if ep % 100 == 1:
            print(f"Epoch {ep}")

            # Calculate likelihood bound
            with torch.no_grad():
                full_loss = miwae_loss(
                    torch.from_numpy(xhat_0).float().to(device),
                    torch.from_numpy(mask_mnar).float().to(device),
                    encoder,
                    decoder,
                    p_z,
                    Config.K_SAMPLES,
                    Config.LATENT_DIM,
                    p,
                    device,
                )
                likelihood_bound = -np.log(Config.K_SAMPLES) - full_loss.cpu().numpy()
                print(f"MIWAE likelihood bound: {likelihood_bound}")

            # Perform imputation
            with torch.no_grad():
                imputed = miwae_impute(
                    torch.from_numpy(xhat_0).float().to(device),
                    torch.from_numpy(mask_mnar).float().to(device),
                    encoder,
                    decoder,
                    p_z,
                    10,
                    Config.LATENT_DIM,
                    p,
                    device,
                )
                xhat[~mask_mnar] = imputed.cpu().numpy()[~mask_mnar]

            err = mse(xhat, original_data_mnar, mask_mnar)
            mse_train.append(err)
            print(f"Imputation MSE: {err}")
            print("-----")

    # Train baselines and plot results
    print("Training baseline methods...")
    baselines = train_baselines(data_nan_mnar, original_data_mnar, mask_mnar)

    print("\nFinal Results:")
    print(f"MIWAE MSE: {mse_train[-1]:.4f}")
    print(f"MissForest MSE: {baselines['missforest'][1]:.4f}")
    print(f"Iterative Ridge MSE: {baselines['ridge'][1]:.4f}")
    print(f"Mean Imputation MSE: {baselines['mean'][1]:.4f}")

    plot_results(mse_train, baselines)


if __name__ == "__main__":
    main()

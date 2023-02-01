import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import wandb


def make_dataloaders(root="./mnist_data/", batch_size=64):
    # MNIST Dataset
    train_dataset = datasets.MNIST(
        root=root, train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root=root, train=False, transform=transforms.ToTensor(), download=False
    )

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train():
    if not os.path.exists("models"):
        os.makedirs("models")

    # Training configuration
    default_config = dict(epochs=15, batch_size=64, dim1=512, optimizer="adam")

    wandb.init(project="mnist-vae", config=default_config)

    # Create dataloaders
    train_loader, test_loader = make_dataloaders(
        batch_size=default_config["batch_size"]
    )

    # Create network
    vae = VAE(
        x_dim=784,
        h_dim1=default_config["dim1"],
        h_dim2=default_config["dim1"] // 2,
        z_dim=2,
    )

    # Define optimizer
    if wandb.config.optimizer == "adam":
        optimizer = optim.Adam(vae.parameters())
    else:
        optimizer = optim.SGD(vae.parameters(), lr=0.01, momentum=0.5)

    wandb.watch(vae, log="all")

    # Generate random samples of latent space
    z = torch.randn(64, 2)

    # Epoch loop
    for epoch in range(default_config["epochs"]):
        # Train 1 epoch
        vae.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data
            optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Log to wandb every 100 batches
            if batch_idx % 10 == 0:
                wandb.log(dict(epoch=epoch, train_loss=loss.item() / len(data)))

        # Test
        vae.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data
                recon, mu, log_var = vae(data)
                test_loss += loss_function(recon, data, mu, log_var).item()
        test_loss /= len(test_loader.dataset)
        wandb.log(dict(test_loss=test_loss))

        # Checkpoint model
        path = f"models/model_ckpt_epoch={epoch}.pt"
        torch.save(vae.state_dict(), path)

        # Create sample predictions
        with torch.no_grad():
            sample = vae.decoder(z)
            collage = sample.view(64, 1, 28, 28)
            wandb.log(dict(samples=wandb.Image(collage)))


if __name__ == "__main__":
    train()

from src.model import Autoencoder
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl


class LitAutoencoder(pl.LightningModule):

    def __init__(self, model, lr: float = 1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *kwargs)
        self.model = model
        self.forward = self.model.forward

    def shared_step(self, batch, mode='train'):
        x, _ = batch
        recon, latent, loss = self(x)
        self.log('{mode}_loss', loss)
        return recon, latent, loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)[-1]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def fit(self, *data, **trainer_kwargs):
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self, *data)


ds = CIFAR10('./', download=True, transform=ToTensor())
loader = DataLoader(ds, batch_size=32, shuffle=True)
model = LitAutoencoder(Autoencoder(input_dim=3072))
model.fit(loader, gpus=1, max_epochs=1)

module_name = model.model.__module__
member_name = model.model.__class__.__qualname__
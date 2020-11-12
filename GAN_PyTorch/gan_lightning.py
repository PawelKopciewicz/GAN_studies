import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch
from GAN import options
from GAN import RootDataSet

import pytorch_lightning as pl

os.makedirs("images", exist_ok=True)

opt = options.options().opt
print(opt)

#img_shape = (opt.channels, opt.img_size, opt.img_size)
Dataset = RootDataSet.RootDataSet("../files/PhaseSpaceSimulation.root")

class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.Generator = nn.Sequential(
            nn.Linear(opt.latent_dim,128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,0.8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, Dataset.leaves),
            nn.Tanh()
        )
        self.Discriminator = nn.Sequential(
            nn.Linear(Dataset.leaves, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, z, setup):
        if setup == "Generator":
            img = self.Generator(z)
            return img
        if setup == "Discriminator":
            validity = self.model(z)  # (img_flat)
            return validity
    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        opt_disc = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        return (opt_gen, opt_disc)
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss



# Loss function
adversarial_loss = torch.nn.BCELoss()

#generator = Generator()
#discriminator = Discriminator()

GenAdvNet = GAN()

dataloader = torch.utils.data.DataLoader(
    Dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(GenAdvNet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i,imgs in enumerate(dataloader):
        #print(imgs)
        #print(q)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        #print(valid)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        #print(real_imgs)

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        #print(z)
        #print(imgs.shape[0])
        # Generate a batch of images
        gen_imgs = GenAdvNet(z, "Generator")
        #print(gen_imgs)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(GenAdvNet(gen_imgs, "Discriminator"), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(GenAdvNet(real_imgs, "Discriminator"), valid)
        fake_loss = adversarial_loss(GenAdvNet(gen_imgs.detach(),"Discriminator"), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)



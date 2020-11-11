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
os.makedirs("images", exist_ok=True)

opt = options.options().opt
print(opt)

#img_shape = (opt.channels, opt.img_size, opt.img_size)
Dataset = RootDataSet.RootDataSet("../files/PhaseSpaceSimulation.root")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
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

    def forward(self, z):
        img = self.model(z)
        #img = img.view(img.size(0), *(1,9))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(Dataset.leaves, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)

dataloader = torch.utils.data.DataLoader(
    Dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        #print(z)
        #print(imgs.shape[0])
        # Generate a batch of images
        gen_imgs = generator(z)
        #print(gen_imgs)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
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




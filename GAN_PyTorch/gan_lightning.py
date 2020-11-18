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

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        opt_disc = torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        return (opt_gen, opt_disc)
    
    def training_step(self, batch, batch_idx, optimizer_idx):

        (opt_gen, opt_disc) = self.optimizers()

        #ground truths
        valid = Variable(torch.FloatTensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(batch.size(0), 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(batch.type(torch.FloatTensor))

        opt_gen.zero_grad()
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch.shape[0], opt.latent_dim))))

        gen_imgs = self.Generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.nn.BCELoss()(self.Discriminator(gen_imgs), valid)

        print('\n',g_loss)
        g_loss.backward()
        opt_gen.step()

        opt_disc.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = torch.nn.BCELoss()(self.Discriminator(real_imgs), valid)

        fake_loss = torch.nn.BCELoss()(self.Discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        opt_disc.step()


adversarial_loss = torch.nn.BCELoss()

GenAdvNet = GAN()

dataloader = torch.utils.data.DataLoader(
    Dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

gan = GAN()
pl.Trainer().fit(gan, dataloader)


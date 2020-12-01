import os
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

opt = options.options().opt
print(opt)


#img_shape = (opt.channels, opt.img_size, opt.img_size)
Dataset = RootDataSet.RootDataSet("../files/PhaseSpaceSimulation.root")

class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.Generator = nn.Sequential(
            nn.Linear(opt.latent_dim,64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64,64),
            nn.BatchNorm1d(64,0.8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, Dataset.leaves),
            nn.Tanh()
        )
        self.Discriminator = nn.Sequential(
            nn.Linear(Dataset.leaves, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64),
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

        #print('\n',g_loss)
        g_loss.backward()
        opt_gen.step()

        opt_disc.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = torch.nn.BCELoss()(self.Discriminator(real_imgs), valid)

        fake_loss = torch.nn.BCELoss()(self.Discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        opt_disc.step()

        #return g_loss

if __name__ == " __main__ ":
    dataloader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=opt.batch_size,
       shuffle=True,
    )

    GenAdvNet = GAN()
    trainer = pl.Trainer(max_epochs=50)

    trainer.fit(GenAdvNet, dataloader)

    os.makedirs("images", exist_ok=True)

    #Size of testing data
    r = 5000

    z_test = Variable(torch.FloatTensor(np.random.normal(0, 1, (r, opt.latent_dim))))
    gen_p = GenAdvNet.Generator(z_test)
    gen_p = gen_p.detach().numpy()


    gen_p = Dataset.scaler.inverse_transform(gen_p)

    Mass_B = np.zeros(r)
    P1x = np.zeros(r)
    P1x_t = np.zeros(r)
    P3z = np.zeros(r)
    P_tot = np.zeros(r)
    E_tot = np.zeros(r)
    Mass_K = 493.677
    data_t = Dataset.data_t

    for i in range(r):
        p_products = np.array([np.sqrt(np.square(gen_p[i][0]) + np.square(gen_p[i][1]) + np.square(gen_p[i][2])),
                               np.sqrt(np.square(gen_p[i][3]) + np.square(gen_p[i][4]) + np.square(gen_p[i][5])),
                               np.sqrt(np.square(gen_p[i][6]) + np.square(gen_p[i][7]) + np.square(gen_p[i][8]))])
        p_total = np.sqrt(np.square(gen_p[i][0] + gen_p[i][3] + gen_p[i][6]) +
                          np.square(gen_p[i][1] + gen_p[i][4] + gen_p[i][7]) +
                          np.square(gen_p[i][2] + gen_p[i][5] + gen_p[i][8]))
        E_total = np.sqrt(np.square(p_products) + Mass_K ** 2)
        Mass_B[i] = math.sqrt(np.sum(E_total) ** 2 - p_total ** 2)
        P1x[i] = gen_p[i][0]
        P1x_t[i] = data_t[i][0]
        P3z[i] = gen_p[i][8]
        P_tot[i] = p_total
        E_tot[i] = np.sum(E_total)


    print(Mass_B)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    n, bins, patches = axs[0, 0].hist(Mass_B, 200, range=(1000, 50000), alpha=0.5, label='Generated data')
    # axs[0,0].hist(Mass_B_data[0:10000], 200, range=(0,50000), alpha=0.5, label = 'Input data')
    axs[0, 0].set_xlabel('Mass of the B meson [MeV]')
    axs[0, 0].set_ylabel('Number of counts')
    #axs[0, 0].axvline(x=np.mean(Mass_B_data), color='r', linestyle='dashed')
    # axs[0,0].legend(loc='upper right')
    #MPV_mass_predicted.append(np.mean(bins[np.where(n == np.amax(n))]))

    n2, bins2, patches2 = axs[0, 1].hist(P1x, 100, range=(-100000, 100000), alpha=0.5, label='Generated data')
    axs[0, 1].hist(P1x_t[0:r], 100, range=(-100000, 100000), label='Input data', alpha=0.5)
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].set_xlabel('Momentum X of K1 [MeV]')
    axs[0, 1].set_ylabel('Number of counts')

    fig.savefig('Pytorch_lighning_testing.png')

    plt.close()


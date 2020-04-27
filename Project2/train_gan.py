'''
train and test GAN model on airfoils
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from dataset import AirfoilDataset
from gan import Discriminator, Generator
from utils import *
import pdb


def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # define dataset and dataloader
    dataset = AirfoilDataset()
    airfoil_x = dataset.get_x()
    airfoil_dim = airfoil_x.shape[0]
    airfoil_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # hyperparameters
    latent_dim = 16 # please do not change latent dimension
    lr_dis = 0.00005 # discriminator learning rate
    lr_gen = 0.00005 # generator learning rate
    num_epochs = 620
    
    # build the model
    dis = Discriminator(input_dim=airfoil_dim).to(device)
    gen = Generator(latent_dim=latent_dim, airfoil_dim=airfoil_dim).to(device)
    print("Distrminator model:\n", dis)
    print("Generator model:\n", gen)

    # define your GAN loss function here
    # you may need to define your own GAN loss function/class
    # loss = ?

    
    criterion = nn.BCELoss()

    # define optimizer for discriminator and generator separately
    optim_dis = Adam(dis.parameters(), lr=lr_dis)
    optim_gen = Adam(gen.parameters(), lr=lr_gen)

    losses_GG = []
    losses_DD = []
    
    # train the GAN model
    for epoch in range(num_epochs):
        losses_D = []
        losses_G = []
        for n_batch, (local_batch, __) in enumerate(airfoil_dataloader):

            y_real = local_batch.to(device)
            
            # train discriminator

            optim_dis.zero_grad()
            outDis_real = dis(y_real)
            label_real  = torch.ones([y_real.shape[0],1]).to(device)
            lossD_real  = criterion(outDis_real, label_real)

            noise  = torch.randn(y_real.shape[0], latent_dim, device=device)
            y_fake = gen(noise)

            outDis_fake = dis(y_fake)
            label_fake  = torch.zeros([y_real.shape[0],1]).to(device)
            lossD_fake  = criterion(outDis_fake, label_fake)

            loss_dis = lossD_real + lossD_fake
            losses_D.append(loss_dis.item())

            # calculate customized GAN loss for discriminator
            
            loss_dis.backward(retain_graph=True)
            optim_dis.step()

            # train generator
            optim_gen.zero_grad()

            outDis_fake = dis(y_fake)
            label_real  = torch.ones([y_real.shape[0],1]).to(device)
            loss_gen  = criterion(outDis_fake, label_real)

            losses_G.append(loss_gen.item())
            
            # pdb.set_trace()

            
            loss_gen.backward()
            optim_gen.step()

            # print loss while training
            if (n_batch + 1) % 30 == 0:
                print("Epoch: [{}/{}], Batch: {}, Discriminator loss: {}, Generator loss: {}".format(
                    epoch, num_epochs, n_batch, loss_dis.item(), loss_gen.item()))
        losses_GG.append(np.mean(np.array(losses_G)))
        losses_DD.append(np.mean(np.array(losses_D)))

    plt.title("Generator Loss")
    plt.ylabel('Loss')
    plt.xlabel('Num of Epochs')
    plt.plot(losses_G)

    plt.show()

    plt.title("Discriminator Loss")
    plt.ylabel('Loss')
    plt.xlabel('Num of Epochs')
    plt.plot(losses_D)

    # test trained GAN model
    num_samples = 100
    # create random noise 
    noise = torch.randn((num_samples, latent_dim)).to(device)
    # generate airfoils
    gen_airfoils = gen(noise)
    if 'cuda' in device:
        gen_airfoils = gen_airfoils.detach().cpu().numpy()
    else:
        gen_airfoils = gen_airfoils.detach().numpy()

    # plot generated airfoils
    plot_airfoils(airfoil_x, gen_airfoils)


if __name__ == "__main__":
    main()


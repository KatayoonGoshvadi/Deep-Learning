import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import FlowDataset
from lstm import FlowLSTM
import pdb
import matplotlib.pyplot as plt


def main():
    # check if cuda available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # define dataset and dataloader
    train_dataset = FlowDataset(mode='train')
    test_dataset = FlowDataset(mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # hyper-parameters
    num_epochs = 10 #20
    lr = 0.001
    input_size = 17 # do not change input size
    hidden_size = 128
    num_layers = 2
    dropout = 0.1


    model = FlowLSTM(
        input_size=input_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()

    # define optimizer for lstm model
    optim = Adam(model.parameters(), lr=lr)
    lossess = []
    for epoch in range(num_epochs):
        losses = []
        for n_batch, (in_batch, label) in enumerate(train_loader):
            in_batch, label = in_batch.to(device), label.to(device)

           # train LSTM

            optim.zero_grad()

            output = model(in_batch)


            loss = criterion(output, label)
            losses.append(loss.item())

            loss.backward()
            optim.step()

            # print loss while training

            if (n_batch + 1) % 200 == 0:
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, n_batch, loss.item()))
        lossess.append(np.mean(np.array(losses)))

    from matplotlib import pyplot as plt
    plt.figure()
    plt.title('LSTM Loss')
    plt.ylabel('Loss')
    plt.xlabel('Num of Epochs')
    plt.plot(lossess)
    plt.show()


    # test trained LSTM model
    l1_err, l2_err = 0, 0
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for n_batch, (in_batch, label) in enumerate(test_loader):
            in_batch, label = in_batch.to(device), label.to(device)
            pred = model.test(in_batch)

            l1_err += l1_loss(pred, label).item()
            l2_err += l2_loss(pred, label).item()

    print("Test L1 error:", l1_err)
    print("Test L2 error:", l2_err)

    # visualize the prediction comparing to the ground truth
    if device is 'cpu':
        pred = pred.detach().numpy()[0,:,:]
        label = label.detach().numpy()[0,:,:]
    else:
        pred = pred.detach().cpu().numpy()[0,:,:]
        label = label.detach().cpu().numpy()[0,:,:]

    r = []
    num_points = 17
    interval = 1./num_points
    x = int(num_points/2)
    for j in range(-x,x+1):
        r.append(interval*j)

    from matplotlib import pyplot as plt
    plt.figure()
    for i in range(1, len(pred)):
        c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
        plt.plot(pred[i], r, label='t = %s' %(i), c=c)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    plt.show()

    plt.figure()
    for i in range(1, len(label)):
        c = (i/(num_points+1), 1-i/(num_points+1), 0.5)
        plt.plot(label[i], r, label='t = %s' %(i), c=c)
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')
    plt.show()


if __name__ == "__main__":
    main()


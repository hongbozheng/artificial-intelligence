# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP3. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        # increase the hidden layer size to 128 to enhance performance
        self.net = nn.Sequential(nn.Linear(in_size,128),nn.ReLU(),nn.Linear(128,out_size))
        # initialize optimizer to be SGD
        # momentum is tuned to 0.70 to reach best performance
        # weight_decay is tuned to 1e-4 to enhance performance
        # turn off nesterov accelerated gradient
        self.optimizer = optim.SGD(self.parameters(),lr=lrate,momentum=0.70,weight_decay=1e-4,nesterov=False)
        #raise NotImplementedError("You need to write this part!")
    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        fx = self.net(x)    # calculate fx (predict y)
        return fx
        #raise NotImplementedError("You need to write this part!")
        #return torch.ones(x.shape[0], 1)

    def step(self,x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimizer.zero_grad()                  # clear optimizer buffer
        x = (x-x.mean())/x.std()                    # normalize features
        y_hat = self.forward(x)                     # compute predict y
        train_loss = self.loss_fn(y_hat,y)          # calculate loss (between y_hat and y)
        train_loss.backward()
        self.optimizer.step()
        return train_loss.detach().cpu().numpy()    # need detach().cpu().numpy() since not return 1 value
        #raise NotImplementedError("You need to write this part!")
        #return 0.0



def fit(train_set,train_labels,dev_set,epochs,batch_size=50):   # batch_size=50 since gradescope uses 50
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # lrate is tuned to 8.0e-3 to enhance performance
    net = NeuralNet(lrate=8.0e-3,loss_fn=nn.CrossEntropyLoss(),in_size=3072,out_size=4)
    train_loss = list()
    y_hats = []

    train_set_batch = get_dataset_from_arrays(train_set, train_labels)
    train_set_data = DataLoader(dataset=train_set_batch, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        for train_batch in train_set_data:
            train_loss.append(net.step(train_batch['features'],train_batch['labels']))

    # Data Standardization
    dev_set_standard = (dev_set-train_set.mean())/train_set.std()

    for i in dev_set_standard:
        fx = net.forward(i)
        y_hats.append(torch.argmax(fx).item())

    yhats = np.array(y_hats)

    return train_loss,yhats,net
    #raise NotImplementedError("You need to write this part!")
    #return [],[],None

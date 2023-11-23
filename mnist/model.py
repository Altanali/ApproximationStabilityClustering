import torch
import torch.nn as nn
import torch.nn.functional as F

# create an autoencoder model with num_layers layers for mnist
class Autoencoder(nn.Module):
    def __init__(self, hidden_size=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(256, 64),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(64, hidden_size),
            #nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y, h

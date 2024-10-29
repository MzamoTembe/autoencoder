import torch
from torch import nn
from typing import List, Tuple


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, dropout=0.2, negative_slope=0.01):
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim

        encoder_layers = []
        in_features = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        encoder_layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_features = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
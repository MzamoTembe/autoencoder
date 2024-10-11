import torch
from torch import nn
from typing import List, Tuple


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        latent_dim: int,
        dropout: float = 0.2,
        negative_slope: float = 0.01
    ) -> None:
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.dropout_rate = dropout
        self.negative_slope = negative_slope

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Sequential:
        layers = []
        in_dim = self.input_dim
        for layer_size in self.hidden_layers:
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
            in_dim = layer_size
        layers.append(nn.Linear(self.hidden_layers[-1], self.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers = []
        in_dim = self.latent_dim
        reversed_layers = list(reversed(self.hidden_layers))
        for layer_size in reversed_layers:
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
            in_dim = layer_size
        layers.append(nn.Linear(reversed_layers[-1], self.input_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_representation = self.encoder(x)
        reconstructed_output = self.decoder(latent_representation)
        return reconstructed_output, latent_representation
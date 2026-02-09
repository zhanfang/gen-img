import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, in_channels, hidden_dims[::-1])
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Encoder layers
        for dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU()
            ))
            in_channels = dim
        
        # Bottleneck
        self.final_layer = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=4, out_channels=3, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Initial layer
        self.initial_layer = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1)
        
        # Decoder layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, hidden_dims[i + 1]),
                nn.SiLU()
            ))
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

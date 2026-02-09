import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[32, 64, 128, 256], text_emb_dim=512):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.time_embedding = TimeEmbedding(features[0])
        self.text_projection = nn.Linear(text_emb_dim, features[0])
        time_emb_dim = features[0]
        
        # Down part
        for feature in features:
            self.downs.append(DownBlock(in_channels, feature, time_emb_dim))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = BottleneckBlock(features[-1], features[-1] * 2, time_emb_dim)
        
        # Up part
        for feature in reversed(features):
            self.ups.append(UpBlock(feature * 2, feature, time_emb_dim))
        
        # Final layer
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x, t, text_emb=None):
        t_emb = self.time_embedding(t)
        
        # Process text embedding if provided
        if text_emb is not None:
            text_emb = self.text_projection(text_emb.mean(dim=1))
            t_emb = t_emb + text_emb
        
        skip_connections = []
        
        # Down part
        for down in self.downs:
            skip, x = down(x, t_emb)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x, t_emb)
        
        # Up part
        skip_connections = skip_connections[::-1]
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[i], t_emb)
        
        # Final layer
        return self.final(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
    
    def forward(self, t):
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=t.device))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        return emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, t_emb):
        # First block
        h = self.norm1(self.conv1(x))
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(h)
        
        # Second block
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        # Downsample
        return h, self.downsample(h)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x, t_emb):
        # First block
        h = self.norm1(self.conv1(x))
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(h)
        
        # Second block
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        return h

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x, skip_x, t_emb):
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip_x], dim=1)
        
        # First block
        h = self.norm1(self.conv1(x))
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(h)
        
        # Second block
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        return h

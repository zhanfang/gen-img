import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    变分自编码器 (VAE) 用于图像压缩和潜在空间表示。
    
    VAE 由两部分组成：
    1. 编码器 (Encoder)：将高维图像压缩为低维潜在向量。
    2. 解码器 (Decoder)：将潜在向量重建回图像。
    
    在 Stable Diffusion 中，扩散过程发生在潜在空间，这大大降低了计算成本。
    """
    def __init__(self, in_channels=3, latent_dim=4, hidden_dims=[64, 128, 256, 512]):
        """
        初始化 VAE 模型。
        
        Args:
            in_channels (int): 输入图像的通道数 (RGB 为 3)。
            latent_dim (int): 潜在空间的通道维度 (例如 SD 中为 4)。
            hidden_dims (list): 编码器/解码器各层的隐藏通道数。
        """
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, in_channels, hidden_dims[::-1])
    
    def encode(self, x):
        """
        将图像编码为潜在表示。
        
        Args:
            x (torch.Tensor): 输入图像张量。
            
        Returns:
            torch.Tensor: 潜在向量。
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        从潜在表示解码回图像。
        
        Args:
            z (torch.Tensor): 潜在向量。
            
        Returns:
            torch.Tensor: 重建的图像张量。
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        前向传播：编码然后解码。
        
        Args:
            x (torch.Tensor): 输入图像。
            
        Returns:
            torch.Tensor: 重建图像。
        """
        z = self.encode(x)
        return self.decode(z)

class Encoder(nn.Module):
    """
    VAE 的编码器部分。
    
    使用一系列卷积层和下采样操作将图像压缩到潜在空间。
    """
    def __init__(self, in_channels=3, latent_dim=4, hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 编码器层 (逐步下采样)
        for dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, dim),
                nn.SiLU()
            ))
            in_channels = dim
        
        # 瓶颈层，映射到最终的潜在维度
        self.final_layer = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

class Decoder(nn.Module):
    """
    VAE 的解码器部分。
    
    使用一系列转置卷积层（上采样）将潜在向量重建回像素空间图像。
    """
    def __init__(self, latent_dim=4, out_channels=3, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 初始层，从潜在维度映射回隐藏维度
        self.initial_layer = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1)
        
        # 解码器层 (逐步上采样)
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, hidden_dims[i + 1]),
                nn.SiLU()
            ))
        
        # 最终层，映射回图像通道并使用 Sigmoid 激活 (将值限制在 [0, 1])
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)

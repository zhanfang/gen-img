import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    用于扩散过程中噪声预测的 UNet 架构。
    
    UNet 接收噪声图像、时间步长和可选的文本嵌入作为输入，并输出预测的噪声。
    它使用带有下采样（编码）和上采样（解码）路径的 U 形架构，通过跳跃连接（Skip Connections）相连。
    
    Attributes:
        downs (nn.ModuleList): 下采样块列表 (DownBlock)。
        ups (nn.ModuleList): 上采样块列表 (UpBlock)。
        time_embedding (TimeEmbedding): 时间步长嵌入模块。
        text_projection (nn.Linear): 文本嵌入投影层。
        bottleneck (BottleneckBlock): UNet 的中间瓶颈块。
        final (nn.Conv2d): 最终的卷积层，用于映射到输出通道。
    """
    def __init__(self, in_channels=4, out_channels=4, features=[32, 64, 128, 256], text_emb_dim=512):
        """
        初始化 UNet 模型。
        
        Args:
            in_channels (int): 输入通道数（例如潜在空间为 4）。
            out_channels (int): 输出通道数（例如预测的噪声为 4）。
            features (list): 每个层级的特征通道数列表。
            text_emb_dim (int): 文本嵌入的维度。
        """
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.time_embedding = TimeEmbedding(features[0])
        self.text_projection = nn.Linear(text_emb_dim, features[0])
        time_emb_dim = features[0]
        
        # 下采样部分 (编码路径)
        for feature in features:
            self.downs.append(DownBlock(in_channels, feature, time_emb_dim))
            in_channels = feature
        
        # 瓶颈部分 (中间部分)
        self.bottleneck = BottleneckBlock(features[-1], features[-1] * 2, time_emb_dim)
        
        # 上采样部分 (解码路径)
        for feature in reversed(features):
            self.ups.append(UpBlock(feature * 2, feature, time_emb_dim))
        
        # 最终层，用于生成输出
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x, t, text_emb=None):
        """
        UNet 的前向传播。
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)。
            t (torch.Tensor): 时间步长张量，形状为 (batch_size,)。
            text_emb (torch.Tensor, optional): 文本嵌入张量，形状为 (batch_size, seq_len, text_emb_dim)。
        
        Returns:
            torch.Tensor: 预测的噪声张量，形状为 (batch_size, out_channels, height, width)。
        """
        # 嵌入时间步长
        t_emb = self.time_embedding(t)
        
        # 如果提供了文本嵌入，则进行处理 (条件生成)
        if text_emb is not None:
            # 简单的投影：对文本嵌入进行平均池化并投影到时间嵌入维度
            # 在完整的 Stable Diffusion 中，这里会使用 Cross-Attention
            text_emb = self.text_projection(text_emb.mean(dim=1))
            t_emb = t_emb + text_emb
        
        skip_connections = []
        
        # 下采样部分：编码并存储跳跃连接
        for down in self.downs:
            skip, x = down(x, t_emb)
            skip_connections.append(skip)
        
        # 瓶颈部分：在最低分辨率下处理
        x = self.bottleneck(x, t_emb)
        
        # 上采样部分：解码并拼接跳跃连接
        skip_connections = skip_connections[::-1]
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[i], t_emb)
        
        # 最终卷积，匹配输出通道
        return self.final(x)

class TimeEmbedding(nn.Module):
    """
    正弦时间嵌入模块。
    
    使用正弦函数后接 MLP 将标量时间步长转换为向量嵌入。
    这使得网络能够理解当前的噪声水平。
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): 嵌入的输出维度。
        """
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
    
    def forward(self, t):
        """
        Args:
            t (torch.Tensor): 时间步长 (batch_size,)。
        
        Returns:
            torch.Tensor: 时间嵌入 (batch_size, dim)。
        """
        # 正弦嵌入计算
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device=t.device))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP 投影
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        return emb

class DownBlock(nn.Module):
    """
    UNet 的下采样块。
    
    由两个带有 GroupNorm 和 SiLU 激活的类 ResNet 卷积块组成，
    后接一个下采样卷积。它还融合了时间嵌入。
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        """
        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            time_emb_dim (int): 时间嵌入的维度。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, t_emb):
        """
        Args:
            x (torch.Tensor): 输入张量。
            t_emb (torch.Tensor): 时间嵌入张量。
            
        Returns:
            tuple: (下采样前的特征, 下采样后的特征)
        """
        # 第一个卷积块
        h = self.norm1(self.conv1(x))
        # 注入时间嵌入
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(h)
        
        # 第二个卷积块
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        # 返回用于跳跃连接的特征和用于下一层的下采样特征
        return h, self.downsample(h)

class BottleneckBlock(nn.Module):
    """
    UNet 的瓶颈块。
    
    UNet 的中间部分，在最低分辨率下运行。
    结构类似于 DownBlock，但没有下采样。
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x, t_emb):
        # 第一个卷积块
        h = self.norm1(self.conv1(x))
        # 注入时间嵌入
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(h)
        
        # 第二个卷积块
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        return h

class UpBlock(nn.Module):
    """
    UNet 的上采样块。
    
    对输入进行上采样，与跳跃连接拼接，并应用卷积块。
    融合了时间嵌入。
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        """
        Args:
            in_channels (int): 输入通道数（将被分割用于跳跃连接拼接）。
            out_channels (int): 输出通道数。
            time_emb_dim (int): 时间嵌入的维度。
        """
        super().__init__()
        # 上采样层 (转置卷积)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x, skip_x, t_emb):
        """
        Args:
            x (torch.Tensor): 上一层的输入张量。
            skip_x (torch.Tensor): 来自编码路径的跳跃连接张量。
            t_emb (torch.Tensor): 时间嵌入张量。
        """
        # 上采样输入
        x = self.upsample(x)
        
        # 沿通道维度与跳跃连接拼接
        x = torch.cat([x, skip_x], dim=1)
        
        # 第一个卷积块
        h = self.norm1(self.conv1(x))
        # 注入时间嵌入
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(h)
        
        # 第二个卷积块
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        return h

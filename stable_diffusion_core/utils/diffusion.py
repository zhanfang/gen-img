import torch
import numpy as np

class DiffusionProcess:
    """
    管理扩散过程（DDPM）。
    
    包括前向扩散过程（向图像添加噪声）和反向采样过程（从噪声中恢复图像）。
    """
    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        """
        初始化扩散过程。
        
        Args:
            num_steps (int): 扩散步数 (T)。
            beta_start (float): beta 调度表的起始值。
            beta_end (float): beta 调度表的结束值。
            device (str): 运行设备。
        """
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # 创建 beta 调度表 (线性)
        self.betas = self._linear_beta_schedule()
        
        # 预计算常用值 (alpha, alpha_cumprod 等)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def _linear_beta_schedule(self):
        """
        生成线性的 beta 调度表。
        """
        return torch.linspace(self.beta_start, self.beta_end, self.num_steps, device=self.device)
    
    def forward_diffusion(self, x_0, t, noise=None):
        """
        前向扩散过程：向图像添加噪声。
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0 (torch.Tensor): 原始图像（或潜在表示）。
            t (torch.Tensor): 时间步长。
            noise (torch.Tensor, optional): 噪声张量。如果不提供，则随机生成。
            
        Returns:
            torch.Tensor: 添加了噪声的图像 x_t。
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def reverse_sample(self, model, x_t, t, text_emb=None):
        """
        反向采样过程：去噪图像。
        p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
        
        Args:
            model (nn.Module): 预测噪声的 UNet 模型。
            x_t (torch.Tensor): 时间步长 t 的噪声图像。
            t (torch.Tensor): 时间步长。
            text_emb (torch.Tensor, optional): 文本嵌入（用于条件生成）。
            
        Returns:
            torch.Tensor: 前一时刻的图像 x_{t-1}。
        """
        with torch.no_grad():
            # 预测噪声
            noise_pred = model(x_t, t, text_emb)
            
            # 计算均值
            # mu_theta = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
            sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t])[:, None, None, None]
            # 注意：这里的维度扩展是为了匹配 batch_size
            
            # 这里需要注意形状匹配，beta 和 cumprod 都是 (T,)，取值后是 (batch,)
            # 需要扩展为 (batch, 1, 1, 1) 以进行广播
            beta_t = self.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            
            mean = sqrt_recip_alphas_t * (x_t - beta_t / sqrt_one_minus_alphas_cumprod_t * noise_pred)
            
            # 添加方差项 (对于 t > 0)
            if t[0] > 0:
                noise = torch.randn_like(x_t)
                variance = self.posterior_variance[t][:, None, None, None]
                return mean + torch.sqrt(variance) * noise
            else:
                return mean
    
    def sample(self, model, batch_size=1, image_size=64, num_channels=4, text_emb=None):
        """
        从纯噪声中采样生成图像（完整的反向过程）。
        
        Args:
            model (nn.Module): UNet 模型。
            batch_size (int): 批量大小。
            image_size (int): 图像（或潜在特征图）大小。
            num_channels (int): 通道数。
            text_emb (torch.Tensor, optional): 文本嵌入。
            
        Returns:
            torch.Tensor: 生成的图像（或潜在表示）。
        """
        # 从标准正态分布采样初始噪声 x_T
        x = torch.randn(batch_size, num_channels, image_size, image_size, device=self.device)
        
        # 逐步去噪：从 T 到 0
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.reverse_sample(model, x, t_tensor, text_emb)
        
        return x
    
    def get_loss(self, model, x_0, noise=None, text_emb=None):
        """
        计算训练损失。
        
        Args:
            model (nn.Module): UNet 模型。
            x_0 (torch.Tensor): 原始图像。
            noise (torch.Tensor, optional): 噪声张量。
            text_emb (torch.Tensor, optional): 文本嵌入。
            
        Returns:
            torch.Tensor: MSE 损失。
        """
        batch_size = x_0.shape[0]
        
        # 随机采样时间步长 t
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device, dtype=torch.long)
        
        # 生成噪声
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # 前向扩散：获取 x_t
        x_t = self.forward_diffusion(x_0, t, noise)
        
        # 模型预测噪声
        noise_pred = model(x_t, t, text_emb)
        
        # 计算预测噪声与真实噪声之间的 MSE 损失
        return torch.nn.functional.mse_loss(noise_pred, noise)

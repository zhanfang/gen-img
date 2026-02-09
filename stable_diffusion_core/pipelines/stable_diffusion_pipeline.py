import torch
from stable_diffusion_core.models.unet import UNet
from stable_diffusion_core.models.vae import VAE
from stable_diffusion_core.models.text_encoder import TextEncoder
from stable_diffusion_core.utils.diffusion import DiffusionProcess
from PIL import Image
import numpy as np

class StableDiffusionPipeline:
    """
    Stable Diffusion 推理管道。
    
    整合了 VAE、UNet、TextEncoder 和 DiffusionProcess，提供一个统一的接口来生成图像。
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化管道并加载各个组件。
        
        Args:
            device (str): 运行设备。
        """
        self.device = device
        
        # 初始化组件
        self.text_encoder = TextEncoder(device=device)
        self.vae = VAE().to(device)
        self.unet = UNet(
            in_channels=4,
            out_channels=4,
            features=[32, 64, 128, 256],
            text_emb_dim=self.text_encoder.get_text_embedding_dim()
        ).to(device)
        self.diffusion = DiffusionProcess(num_steps=1000, device=device)
    
    def load_weights(self, unet_path=None, vae_path=None):
        """
        加载预训练权重。
        
        Args:
            unet_path (str, optional): UNet 权重文件路径。
            vae_path (str, optional): VAE 权重文件路径。
        """
        if unet_path:
            self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
        if vae_path:
            self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
    
    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5, image_size=(64, 64)):
        """
        根据文本提示生成图像。
        
        Args:
            prompt (str): 文本提示。
            num_inference_steps (int): 推理步数。
            guidance_scale (float): 无分类器引导 (Classifier-Free Guidance) 的比例。
                                   (注：当前简化版实现尚未完全集成 CFG，仅作占位)
            image_size (tuple): 生成图像的尺寸 (高度, 宽度)。
            
        Returns:
            PIL.Image.Image: 生成的图像。
        """
        # 编码文本
        text_emb = self.text_encoder.encode([prompt])
        
        # 设置推理步数并重新计算扩散参数
        # 注意：这里是为了支持推理时动态调整步数
        self.diffusion.num_steps = num_inference_steps
        self.diffusion.betas = self.diffusion._linear_beta_schedule()
        self.diffusion.alphas = 1.0 - self.diffusion.betas
        self.diffusion.alphas_cumprod = torch.cumprod(self.diffusion.alphas, dim=0)
        self.diffusion.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.diffusion.alphas_cumprod[:-1]])
        self.diffusion.sqrt_alphas_cumprod = torch.sqrt(self.diffusion.alphas_cumprod)
        self.diffusion.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.diffusion.alphas_cumprod)
        self.diffusion.posterior_variance = self.diffusion.betas * (1.0 - self.diffusion.alphas_cumprod_prev) / (1.0 - self.diffusion.alphas_cumprod)
        
        # 从扩散模型采样 (在潜在空间)
        # 注意：image_size 需要除以 VAE 的下采样率 (通常是 8)
        latent = self.diffusion.sample(
            self.unet,
            batch_size=1,
            image_size=image_size[0] // 8,  # VAE 下采样 8 倍
            num_channels=4,
            text_emb=text_emb
        )
        
        # 将潜在表示解码为图像
        image = self.vae.decode(latent)
        
        # 转换为 PIL 图像
        image = self._tensor_to_pil(image)
        
        return image
    
    def _tensor_to_pil(self, tensor):
        """
        将 PyTorch 张量转换为 PIL 图像。
        
        Args:
            tensor (torch.Tensor): 形状为 (1, 3, H, W) 的张量，值范围 [0, 1]。
            
        Returns:
            PIL.Image.Image: PIL 图像对象。
        """
        tensor = tensor.squeeze(0).cpu().detach()
        tensor = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
        tensor = tensor.transpose(1, 2, 0)
        return Image.fromarray(tensor)

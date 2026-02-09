import torch
from stable_diffusion_core.models.unet import UNet
from stable_diffusion_core.models.vae import VAE
from stable_diffusion_core.models.text_encoder import TextEncoder
from stable_diffusion_core.utils.diffusion import DiffusionProcess
from PIL import Image
import numpy as np

class StableDiffusionPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize components
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
        Load pre-trained weights
        unet_path: path to UNet weights
        vae_path: path to VAE weights
        """
        if unet_path:
            self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
        if vae_path:
            self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
    
    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5, image_size=(64, 64)):
        """
        Generate image from prompt
        prompt: text prompt
        num_inference_steps: number of inference steps
        guidance_scale: guidance scale for classifier-free guidance
        image_size: image size
        """
        # Encode text
        text_emb = self.text_encoder.encode([prompt])
        
        # Set number of inference steps
        self.diffusion.num_steps = num_inference_steps
        self.diffusion.betas = self.diffusion._linear_beta_schedule()
        self.diffusion.alphas = 1.0 - self.diffusion.betas
        self.diffusion.alphas_cumprod = torch.cumprod(self.diffusion.alphas, dim=0)
        self.diffusion.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.diffusion.alphas_cumprod[:-1]])
        self.diffusion.sqrt_alphas_cumprod = torch.sqrt(self.diffusion.alphas_cumprod)
        self.diffusion.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.diffusion.alphas_cumprod)
        self.diffusion.posterior_variance = self.diffusion.betas * (1.0 - self.diffusion.alphas_cumprod_prev) / (1.0 - self.diffusion.alphas_cumprod)
        
        # Sample from diffusion model
        latent = self.diffusion.sample(
            self.unet,
            batch_size=1,
            image_size=image_size[0] // 8,  # VAE downsamples by 8x
            num_channels=4,
            text_emb=text_emb
        )
        
        # Decode latent to image
        image = self.vae.decode(latent)
        
        # Convert to PIL image
        image = self._tensor_to_pil(image)
        
        return image
    
    def _tensor_to_pil(self, tensor):
        """
        Convert tensor to PIL image
        tensor: (1, 3, H, W) tensor in [0, 1]
        """
        tensor = tensor.squeeze(0).cpu().detach()
        tensor = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
        tensor = tensor.transpose(1, 2, 0)
        return Image.fromarray(tensor)

import torch
import numpy as np

class DiffusionProcess:
    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Create beta schedule
        self.betas = self._linear_beta_schedule()
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def _linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_steps, device=self.device)
    
    def forward_diffusion(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to image
        x_0: original image
        t: timestep
        noise: optional noise tensor
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def reverse_sample(self, model, x_t, t):
        """
        Reverse sampling process: denoise image
        model: UNet model
        x_t: noisy image at timestep t
        t: timestep
        """
        with torch.no_grad():
            # Predict noise
            noise_pred = model(x_t, t)
            
            # Compute coefficients
            sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t])
            mean = sqrt_recip_alphas_t * (x_t - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * noise_pred)
            
            # Add noise for t > 0
            if t[0] > 0:
                noise = torch.randn_like(x_t)
                variance = self.posterior_variance[t]
                return mean + torch.sqrt(variance)[:, None, None, None] * noise
            else:
                return mean
    
    def sample(self, model, batch_size=1, image_size=64, num_channels=4, text_emb=None):
        """
        Sample from the model
        model: UNet model
        batch_size: batch size
        image_size: image size
        num_channels: number of channels
        text_emb: text embedding (optional)
        """
        x = torch.randn(batch_size, num_channels, image_size, image_size, device=self.device)
        
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.reverse_sample(model, x, t_tensor, text_emb)
        
        return x
    
    def reverse_sample(self, model, x_t, t, text_emb=None):
        """
        Reverse sampling process: denoise image
        model: UNet model
        x_t: noisy image at timestep t
        t: timestep
        text_emb: text embedding (optional)
        """
        with torch.no_grad():
            # Predict noise
            noise_pred = model(x_t, t, text_emb)
            
            # Compute coefficients
            sqrt_recip_alphas_t = 1.0 / torch.sqrt(self.alphas[t])
            mean = sqrt_recip_alphas_t * (x_t - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * noise_pred)
            
            # Add noise for t > 0
            if t[0] > 0:
                noise = torch.randn_like(x_t)
                variance = self.posterior_variance[t]
                return mean + torch.sqrt(variance)[:, None, None, None] * noise
            else:
                return mean
    
    def get_loss(self, model, x_0, noise=None, text_emb=None):
        """
        Compute loss for training
        model: UNet model
        x_0: original image
        noise: optional noise tensor
        text_emb: text embedding (optional)
        """
        batch_size = x_0.shape[0]
        
        # Random timestep
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device, dtype=torch.long)
        
        # Generate noise
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.forward_diffusion(x_0, t, noise)
        
        # Predict noise
        noise_pred = model(x_t, t, text_emb)
        
        # Compute MSE loss
        return torch.nn.functional.mse_loss(noise_pred, noise)

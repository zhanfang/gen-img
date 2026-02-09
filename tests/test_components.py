import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_diffusion_core.models.unet import UNet
from stable_diffusion_core.models.vae import VAE
from stable_diffusion_core.models.text_encoder import TextEncoder
from stable_diffusion_core.utils.diffusion import DiffusionProcess
from stable_diffusion_core.pipelines.stable_diffusion_pipeline import StableDiffusionPipeline

def test_unet():
    print("Testing UNet...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet().to(device)
    
    # Test forward pass
    x = torch.randn(1, 4, 64, 64, device=device)
    t = torch.randint(0, 1000, (1,), device=device)
    output = model(x, t)
    assert output.shape == (1, 4, 64, 64)
    print("✓ UNet test passed")

def test_vae():
    print("Testing VAE...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VAE().to(device)
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512, device=device)
    output = model(x)
    assert output.shape == (1, 3, 512, 512)
    print("✓ VAE test passed")

def test_text_encoder():
    print("Testing Text Encoder...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = TextEncoder(device=device)
    
    # Test encoding
    text = ["a cat sitting on a bench"]
    embeddings = encoder.encode(text)
    assert embeddings.shape == (1, 77, 512)
    print("✓ Text Encoder test passed")

def test_diffusion_process():
    print("Testing Diffusion Process...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion = DiffusionProcess(device=device)
    
    # Test forward diffusion
    x_0 = torch.randn(1, 4, 64, 64, device=device)
    t = torch.randint(0, 1000, (1,), device=device)
    x_t = diffusion.forward_diffusion(x_0, t)
    assert x_t.shape == (1, 4, 64, 64)
    print("✓ Diffusion Process test passed")

def test_pipeline():
    print("Testing Pipeline...")
    pipeline = StableDiffusionPipeline()
    print("✓ Pipeline initialization test passed")

def run_all_tests():
    print("Running all tests...")
    test_unet()
    test_vae()
    test_text_encoder()
    test_diffusion_process()
    test_pipeline()
    print("\nAll tests passed! 🎉")

if __name__ == "__main__":
    run_all_tests()

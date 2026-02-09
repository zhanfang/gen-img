# Stable Diffusion Core Implementation

This is a learning project that implements the core components of Stable Diffusion, a text-to-image diffusion model.

📖 **[Read the Core Principle Documentation](docs/stable_diffusion_principle.md)** (Chinese) - A detailed explanation of how Stable Diffusion works and how this project implements it.

## Project Structure

```
stable_diffusion/
├── stable_diffusion_core/
│   ├── models/
│   │   ├── unet.py          # UNet model for noise prediction
│   │   ├── vae.py            # VAE for latent space conversion
│   │   └── text_encoder.py   # Text encoder using CLIP
│   ├── utils/
│   │   └── diffusion.py      # Diffusion process implementation
│   └── pipelines/
│       └── stable_diffusion_pipeline.py  # Complete inference pipeline
├── examples/
│   └── generate_image.py     # Example script for image generation
├── requirements.txt          # Dependencies
├── setup.py                  # Project setup
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- PyTorch
- TorchVision
- Transformers
- Diffusers
- NumPy
- Pillow
- tqdm

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Usage

### Generate an image from text prompt

```bash
python examples/generate_image.py "a cat sitting on a bench"
```

### Using the pipeline directly

```python
from stable_diffusion_core.pipelines.stable_diffusion_pipeline import StableDiffusionPipeline

# Initialize pipeline
pipeline = StableDiffusionPipeline()

# Generate image
image = pipeline.generate(
    prompt="a cat sitting on a bench",
    num_inference_steps=50,
    guidance_scale=7.5,
    image_size=(512, 512)
)

# Save image
image.save("output.png")
```

## Components

### 1. UNet

The UNet model is used to predict noise in the latent space. It consists of:
- Downsampling blocks
- Bottleneck block
- Upsampling blocks
- Time embedding
- Text embedding projection

### 2. VAE (Variational Autoencoder)

The VAE is used to convert between pixel space and latent space:
- Encoder: compresses images to latent space
- Decoder: decompresses latent space to images

### 3. Text Encoder

Uses CLIP to encode text prompts into embeddings that guide the image generation process.

### 4. Diffusion Process

Implements:
- Forward diffusion (adding noise)
- Reverse diffusion (removing noise)
- Sampling from the model

## Note

This implementation is a simplified version of Stable Diffusion designed for educational purposes. It may not produce results as high-quality as the original Stable Diffusion model, but it covers all the core concepts and components.

To use pre-trained weights, you would need to train the models yourself or obtain compatible weights from other sources.

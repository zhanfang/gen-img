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

### ⚠️ Important Note About Output
**Since this project implements the model structure from scratch for learning purposes, the weights are RANDOMLY INITIALIZED.**
**Therefore, the output of `examples/generate_image.py` will be RANDOM NOISE.**

To see real generated images, you need to use pre-trained weights. We provide a script to demonstrate this using the official `diffusers` library.

### 1. Run Custom Implementation (Random Noise)
This verifies that the pipeline code runs correctly end-to-end.

```bash
python examples/generate_image.py "a cat sitting on a bench"
```

### 2. Run with Pre-trained Weights (Real Image)
This uses the `diffusers` library to download and run `runwayml/stable-diffusion-v1-5`.

```bash
python examples/generate_with_diffusers.py "a cat sitting on a bench"
```

### 3. Run Training Demo (Loss Decrease Visualization)
This script trains the model on a single image ("overfitting") to demonstrate the learning capability.
It will generate `target_image.png` (Goal), `output_before_train.png` (Noise), and `output_after_train.png` (Learned).

```bash
python examples/train_demo.py
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

For detailed documentation of each component, please refer to:
- **[UNet Architecture](docs/unet.md)**
- **[VAE (Variational Autoencoder)](docs/vae.md)**
- **[Text Encoder](docs/text_encoder.md)**
- **[Scheduler & Diffusion Process](docs/scheduler.md)**

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

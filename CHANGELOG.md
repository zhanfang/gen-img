# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-09

### Added
- Initial project structure and configuration (`setup.py`, `requirements.txt`).
- **Core Models**:
  - `UNet`: Implemented UNet with time embedding and text conditioning (Cross-Attention/Projection).
  - `VAE`: Implemented Variational Autoencoder for latent space compression and reconstruction.
  - `TextEncoder`: Integrated CLIP Text Model (randomly initialized for learning purposes).
- **Diffusion Process**:
  - Implemented forward diffusion (noise addition) and reverse sampling (denoising).
  - Supported linear beta schedule.
- **Pipeline**:
  - `StableDiffusionPipeline`: Integrated VAE, UNet, and Text Encoder into a complete inference pipeline.
- **Documentation**:
  - Added comprehensive `README.md`.
  - Added detailed Principle Documentation (`docs/stable_diffusion_principle.md`) explaining LDM, UNet, and Diffusion process.
- **Examples**:
  - Added `examples/generate_image.py` script for testing image generation.
- **Tests**:
  - Added unit tests for all core components in `tests/test_components.py`.
- **Utils**:
  - Added `.gitignore` for proper version control.

### Fixed
- Fixed `UNet` dimension mismatch issues in `DownBlock` and `UpBlock`.
- Fixed `TextEncoder` compatibility issue with `transformers` library by using `CLIPTextConfig`.
- Fixed package discovery by adding `__init__.py` files.

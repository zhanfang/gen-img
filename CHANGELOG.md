# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Created `docs/training_demo_explained.md`: detailed documentation explaining training demo principles, debugging process, and key technical solutions.
- Added `temp/` to `.gitignore` and updated scripts to save generated images there.

### Changed
- **Optimization & Fixes**:
  - **VAE**: Replaced `ConvTranspose2d` with `Upsample` + `Conv2d` in Decoder to eliminate checkerboard artifacts.
  - **Pipeline**: Fixed `_tensor_to_pil` logic to correctly map `[-1, 1]` tensor to image, resolving color distortion issues.
  - **Training Demo**:
    - Replaced random VAE with `MockVAE` (using simple interpolation) to ensure meaningful latent representation.
    - Removed VAE scaling factor to improve signal-to-noise ratio for faster convergence.
    - Aligned inference steps (1000) with training schedule to fix "noise veil" issue.
    - Updated script to output files to `temp/` directory.
    - Implemented "Static Thresholding" (Clipping) in Diffusion sampling to prevent numerical explosion and black images.
    - **Performance Boost**: Increased UNet capacity (channels: 64->128->256->512) and added CosineAnnealingLR scheduler, reducing final loss from ~0.02 to ~0.001.
- **Documentation**:
  - Created `docs/development_log.md`: A comprehensive narrative of the debugging and optimization journey.
  - Enhanced component documentation with more detailed Chinese comments.

### Added
- 为核心组件（UNet, VAE, TextEncoder, DiffusionProcess, StableDiffusionPipeline）添加了详细的中文文档和注释。
- 完善了代码结构和说明，方便学习和理解。

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

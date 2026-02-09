# 变分自编码器 (Variational Autoencoder, VAE)

## 概述

VAE 是 Stable Diffusion 中的关键组件，负责将图像在**像素空间 (Pixel Space)** 和**潜在空间 (Latent Space)** 之间进行转换。

Stable Diffusion 被称为**潜在扩散模型 (Latent Diffusion Model, LDM)**，原因正是因为扩散过程不是直接在原始高分辨率图像上进行的，而是在压缩后的潜在空间中进行的。这大大降低了计算复杂度，使得在消费级 GPU 上生成高分辨率图像成为可能。

## 核心结构

VAE 由两部分组成：

1.  **编码器 (Encoder, $\mathcal{E}$)**
2.  **解码器 (Decoder, $\mathcal{D}$)**

### 1. 编码器 (Encoder)

编码器的作用是将输入图像 $x$ 压缩为潜在表示 $z$。
公式表示为：$z = \mathcal{E}(x)$

*   **输入**: 原始图像，形状通常为 `(Batch, 3, H, W)`。
*   **过程**: 通过一系列卷积层和下采样操作，逐步提取特征并降低空间分辨率。
*   **输出**: 潜在向量，形状通常为 `(Batch, 4, H/8, W/8)`。
    *   在标准的 Stable Diffusion 中，下采样率通常为 8。例如，512x512 的图像被压缩为 64x64 的潜在特征图。
    *   通道数通常从 3 (RGB) 变为 4。

### 2. 解码器 (Decoder)

解码器的作用是将潜在表示 $z$ 重建回图像 $\tilde{x}$。
公式表示为：$\tilde{x} = \mathcal{D}(z)$

*   **输入**: 潜在向量，形状为 `(Batch, 4, H/8, W/8)`。
*   **过程**: 通过一系列卷积层和上采样操作，逐步恢复空间分辨率。
*   **输出**: 重建图像，形状为 `(Batch, 3, H, W)`。

## 为什么需要 VAE？

1.  **计算效率**: 在 64x64 的潜在空间上进行扩散去噪，比在 512x512 的像素空间上快得多，显存占用也更低。
2.  **特征保留**: VAE 被训练为尽可能保留图像的语义和感知细节，使得重建后的图像与原图差异很小。
3.  **平滑性**: 潜在空间通常比像素空间更平滑，更有利于扩散模型学习数据的分布。

## 代码实现细节

在本项目 `stable_diffusion_core/models/vae.py` 中，我们实现了一个简化的 VAE 结构：

*   **Encoder**: 使用卷积层 (`nn.Conv2d`) 和组归一化 (`nn.GroupNorm`)，配合 `stride=2` 进行下采样。
*   **Decoder**: 使用转置卷积层 (`nn.ConvTranspose2d`) 进行上采样。
*   **激活函数**: 使用 `SiLU` (Sigmoid Linear Unit)。

```python
# 示例：编码过程
z = vae.encode(image)

# 示例：解码过程
reconstructed_image = vae.decode(z)
```

## 训练目标

虽然本项目侧重于推理，但了解 VAE 的训练目标很重要。VAE 的损失函数通常包括：
1.  **重建损失 (Reconstruction Loss)**: 保证 $x$ 和 $\tilde{x}$ 尽可能相似（通常使用 L1 或 L2 损失，以及感知损失 Perceptual Loss）。
2.  **正则化损失 (Regularization Loss)**: 保证潜在空间 $z$ 符合标准正态分布（使用 KL 散度）。这防止了潜在空间出现“空洞”，使得采样更加稳定。

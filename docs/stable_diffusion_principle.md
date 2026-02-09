# Stable Diffusion 核心原理详解

Stable Diffusion 是一种基于潜在扩散模型（Latent Diffusion Model, LDM）的文本到图像生成模型。它的核心思想不是直接在像素空间处理图像，而是在压缩的潜在空间（Latent Space）中进行扩散过程，从而极大地降低了计算成本并提高了生成质量。

本文档将详细介绍 Stable Diffusion 的核心组件和工作原理，并将其映射到本项目的代码实现中。

## 1. 整体架构

Stable Diffusion 主要由三个核心组件组成：

1.  **变分自编码器 (VAE)**：负责将图像从像素空间压缩到潜在空间，以及从潜在空间恢复到像素空间。
2.  **UNet**：核心的去噪网络，在潜在空间中预测噪声。
3.  **文本编码器 (Text Encoder)**：将文本提示转换为向量表示，通过 Cross-Attention 机制引导图像生成。

## 2. 核心组件详解

### 2.1 像素空间 vs 潜在空间 (Pixel Space vs Latent Space)

传统的扩散模型直接在像素空间操作，对于高分辨率图像，计算量巨大。Stable Diffusion 引入了 **感知压缩 (Perceptual Compression)**：

*   **Encoder (编码器)**：将输入图像 $x$ (例如 $3 \times 512 \times 512$) 压缩为潜在向量 $z$ (例如 $4 \times 64 \times 64$)。
*   **Decoder (解码器)**：将潜在向量 $z$ 重建回图像 $x'$。

**代码对应**：
*   [vae.py](file:///Users/bytedance/Documents/code/github/stable_diffusion/stable_diffusion_core/models/vae.py): 实现了 VAE 的 Encoder 和 Decoder。

### 2.2 扩散过程 (Diffusion Process)

扩散过程包含两个方向：前向扩散和反向扩散。

#### 前向扩散 (Forward Diffusion) - 加噪
这是一个固定的马尔可夫链过程，逐渐向潜在向量 $z_0$ 添加高斯噪声，直到变成纯噪声 $z_T$。

$$q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t} z_{t-1}, \beta_t I)$$

其中 $\beta_t$ 是预定义的方差调度（Schedule）。

**代码对应**：
*   [diffusion.py](file:///Users/bytedance/Documents/code/github/stable_diffusion/stable_diffusion_core/utils/diffusion.py) 中的 `forward_diffusion` 方法。

#### 反向扩散 (Reverse Diffusion) - 去噪
这是生成过程。我们要训练一个神经网络（UNet）来预测并去除噪声，从而从纯噪声 $z_T$ 逐步恢复出有意义的潜在向量 $z_0$。

$$p_\theta(z_{t-1} | z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t))$$

实际上，UNet 预测的是当前时刻的噪声 $\epsilon_\theta(z_t, t)$。

**代码对应**：
*   [diffusion.py](file:///Users/bytedance/Documents/code/github/stable_diffusion/stable_diffusion_core/utils/diffusion.py) 中的 `reverse_sample` 和 `sample` 方法。

### 2.3 UNet 与条件机制 (Conditioning)

UNet 是 Stable Diffusion 的引擎。它接收三个输入：
1.  **Noisy Latent ($z_t$)**：当前时刻带噪声的潜在向量。
2.  **Time Embedding ($t$)**：告诉网络当前处于扩散过程的哪一步（噪声水平）。
3.  **Text Embedding ($c$)**：文本提示的向量表示，用于控制生成内容。

**Cross-Attention 机制**：
文本信息通过 Cross-Attention 层注入到 UNet 的各个层级中。这使得模型能够关注文本描述的关键部分，并在图像的相应位置生成对应内容。

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d}})V$$

其中：
*   $Q$ (Query) 来自图像特征。
*   $K$ (Key) 和 $V$ (Value) 来自文本嵌入。

**代码对应**：
*   [unet.py](file:///Users/bytedance/Documents/code/github/stable_diffusion/stable_diffusion_core/models/unet.py): 实现了 UNet 结构。
*   注意 `text_projection` 和 `forward` 方法中如何处理 `text_emb`。*注：本项目为简化实现，使用了简单的加法/投影注入，标准 SD 使用 Cross-Attention。*

### 2.4 文本编码器 (Text Encoder)

使用预训练的 CLIP (Contrastive Language-Image Pre-training) 模型将文本转换为固定维度的嵌入向量。CLIP 的特点是它理解文本和图像之间的语义关系。

**代码对应**：
*   [text_encoder.py](file:///Users/bytedance/Documents/code/github/stable_diffusion/stable_diffusion_core/models/text_encoder.py): 封装了 Hugging Face 的 CLIP 模型。

## 3. 推理流程 (Inference Pipeline)

完整的生成过程如下：

1.  **文本编码**：将用户输入的 Prompt 输入 Text Encoder，得到文本嵌入 $c$。
2.  **生成初始噪声**：在潜在空间采样纯高斯噪声 $z_T$。
3.  **迭代去噪**：
    *   从 $t=T$ 到 $t=1$ 进行循环。
    *   UNet 预测当前噪声：$\epsilon_{pred} = UNet(z_t, t, c)$。
    *   根据调度器算法（如 DDPM, DDIM, PNDM）计算 $z_{t-1}$（去除一部分噪声）。
4.  **图像重建**：循环结束后得到纯净的潜在向量 $z_0$，将其输入 VAE Decoder，得到最终像素图像 $x'$。

**代码对应**：
*   [stable_diffusion_pipeline.py](file:///Users/bytedance/Documents/code/github/stable_diffusion/stable_diffusion_core/pipelines/stable_diffusion_pipeline.py): 实现了 `generate` 方法，串联了上述所有步骤。

## 4. 总结

Stable Diffusion 的成功在于有效地结合了三个强大的技术：
1.  **LDM**：通过在潜在空间操作解决了高分辨率生成的计算瓶颈。
2.  **UNet + Attention**：强大的去噪能力加上灵活的条件控制。
3.  **CLIP**：提供了丰富的语义理解，使得“以文生图”成为可能。

本项目通过简化代码实现了上述核心逻辑，旨在帮助开发者深入理解其底层工作原理。

## 5. 组件详细文档

为了更深入地了解每个组件的内部工作原理和代码实现，请参考以下详细文档：

*   **[VAE 详解](vae.md)**: 深入了解变分自编码器、潜在空间压缩和重建。
*   **[UNet 详解](unet.md)**: 深入了解 UNet 架构、下采样/上采样路径、时间嵌入和文本条件。
*   **[文本编码器详解](text_encoder.md)**: 深入了解 CLIP 模型、Tokenization 和文本嵌入。
*   **[调度器与扩散过程详解](scheduler.md)**: 深入了解 DDPM、前向扩散公式和反向采样算法。

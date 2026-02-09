# 扩散调度器 (Scheduler / Diffusion Process)

## 概述

调度器（Scheduler）控制着扩散模型中的**噪声添加**（前向过程）和**噪声去除**（反向过程）。它定义了噪声在每一步是如何变化的。

在本项目中，我们实现了一个基础的 **DDPM (Denoising Diffusion Probabilistic Models)** 调度器。

## 两个核心过程

### 1. 前向扩散过程 (Forward Diffusion Process)

*   **目标**: 将真实图像 $x_0$ 逐步破坏，直到变成纯高斯噪声 $x_T$。
*   **公式**: $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$
*   **特性**: 这是一个马尔可夫链。好消息是，我们可以直接采样任意时间步 $t$ 的图像，而不需要一步步计算：
    $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
    其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。

    *   $\beta_t$: 噪声方差调度（每一步加多少噪声）。
    *   $\alpha_t = 1 - \beta_t$
    *   $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ (Alpha Cumprod)

### 2. 反向采样过程 (Reverse Sampling Process)

*   **目标**: 从纯噪声 $x_T$ 开始，一步步去除噪声，恢复出图像 $x_0$。
*   **挑战**: 我们不知道真实的逆转换 $q(x_{t-1} | x_t)$。
*   **解决**: 我们训练一个神经网络（UNet）来预测噪声 $\epsilon_\theta(x_t, t)$，从而近似这个逆转换。
*   **去噪公式**:
    $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
    
    简单来说：
    `上一时刻图像 = (当前图像 - 预测的噪声 * 系数) / 缩放系数 + 随机扰动`

## 关键参数

*   **Timesteps ($T$)**: 扩散的总步数。通常训练时为 1000。
*   **Beta Schedule ($\beta_t$)**: 控制每一步添加/去除噪声的量。
    *   **Linear Schedule**: $\beta$ 从 `beta_start` 到 `beta_end` 线性增加。
    *   **Cosine Schedule**: 使用余弦函数，通常能产生更好的效果。

## 代码实现

在 `stable_diffusion_core/utils/diffusion.py` 中：

*   **初始化**: 预计算 $\beta, \alpha, \bar{\alpha}$ 等参数。
*   **`forward_diffusion`**: 实现 $x_t$ 的直接采样公式。用于训练时构造输入。
*   **`reverse_sample`**: 实现去噪公式。用于推理（生成）时。
*   **`sample`**: 完整的生成循环，从 $t=T$ 迭代到 $t=0$。

```python
# 示例：生成循环
x = torch.randn(...) # 从纯噪声开始
for t in reversed(range(num_steps)):
    # 1. UNet 预测噪声
    noise_pred = unet(x, t, text_emb)
    
    # 2. 根据公式去除一部分噪声
    x = diffusion.reverse_sample(x, noise_pred, t)
```

## 进阶调度器

虽然本项目使用了基础的 DDPM，但 Stable Diffusion 生态中有许多更高级的调度器（如 DDIM, PNDM, Euler Discrete, DPM++），它们的主要目标是**加速采样**，即用更少的步数（如 20-50 步）生成高质量图像。

import torch
import torch.optim as optim
from PIL import Image, ImageDraw
import numpy as np
from stable_diffusion_core.pipelines.stable_diffusion_pipeline import StableDiffusionPipeline
import os

"""
Stable Diffusion 训练演示脚本 (Training Demo)

本脚本展示了 Latent Diffusion Model (LDM) 的核心训练流程。
为了方便理解，我们简化了场景：让模型“过拟合”在一张简单的图片上。

训练流程详解：
1. **数据准备**：创建一张红色的圆圈图片 (RGB)，归一化到 [-1, 1]。
2. **VAE 编码**：使用 VAE Encoder 将像素空间 (Pixel Space) 的图片压缩到潜空间 (Latent Space)。
   - 输入: (3, 256, 256) -> 输出: (4, 16, 16)
   - 目的: 减少计算量，让模型关注语义特征而非像素细节。
3. **文本编码**：将提示词 "a red circle" 转换为文本向量 (Text Embeddings)。
4. **训练循环 (Training Loop)**：
   - **加噪 (Forward Diffusion)**：在潜向量上叠加随机高斯噪声。噪声强度由时间步 t 决定。
   - **预测 (UNet Prediction)**：UNet 接收 [加噪后的潜向量, 时间步 t, 文本向量]，尝试预测**添加的噪声**。
   - **计算 Loss**：比较 [预测噪声] 和 [真实噪声] 的均方误差 (MSE)。
   - **反向传播**：更新 UNet 的参数，使其更准确地预测噪声。
5. **生成验证**：使用训练好的 UNet，从纯噪声开始，通过反向扩散过程生成图片。
"""

def create_target_image():
    """创建一个 256x256 的红底白圈图片作为训练目标"""
    # 256x256, 白色背景
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    # 在中心画一个红色的圆
    draw.ellipse([64, 64, 192, 192], fill='red', outline='red')
    return img

def train_step(pipeline, optimizer, x_0, text_emb):
    """
    执行单步训练
    
    参数:
        pipeline: 包含 UNet, VAE, Scheduler 的管道
        optimizer: 优化器
        x_0: 原始潜向量 (Latent), shape (Batch, 4, H/8, W/8)
        text_emb: 文本向量, shape (Batch, Seq, Dim)
    """
    pipeline.unet.train()
    optimizer.zero_grad()
    
    # === 核心训练逻辑 ===
    # pipeline.diffusion.get_loss 内部做了以下事情：
    # 1. 随机采样时间步 t (0 ~ T)
    # 2. 生成随机噪声 epsilon
    # 3. 计算加噪后的潜向量 x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
    # 4. 让 UNet 预测噪声: pred_epsilon = UNet(x_t, t, text_emb)
    # 5. 计算 MSE Loss: (epsilon - pred_epsilon)^2
    
    loss = pipeline.diffusion.get_loss(pipeline.unet, x_0, text_emb=text_emb)
    
    loss.backward()
    optimizer.step()
    return loss.item()

import torch.nn.functional as F

class MockVAE:
    """
    用于演示的简化版 VAE。
    真实的 VAE 需要单独训练才能将图片编码为有意义的 Latent。
    为了在不加载预训练权重的情况下演示，我们使用简单的下采样/上采样来模拟 VAE 的压缩过程。
    这样可以保证 Latent 具有语义信息（虽然只是模糊的图像）。
    """
    def __init__(self, device):
        self.device = device
    
    def encode(self, x):
        # x: (B, 3, 256, 256)
        # 1. 下采样 8 倍 -> (B, 3, 32, 32)
        x_small = F.interpolate(x, scale_factor=1/8, mode='bilinear', align_corners=False)
        # 2. 扩展通道 3 -> 4 (为了匹配 UNet 输入)
        # 我们简单地复制第一个通道作为第4个通道
        extra_channel = x_small[:, 0:1, :, :]
        latent = torch.cat([x_small, extra_channel], dim=1) # (B, 4, 32, 32)
        # 移除缩放因子，保持 Latent 在 [-1, 1] 范围，提高信噪比 (Signal-to-Noise Ratio)
        # 在这个简易演示中，保留原始幅值能让 UNet 更容易学习
        return latent 
        
    def decode(self, z):
        # z: (B, 4, 32, 32)
        # z = z / 0.18215 # 对应移除 encode 中的缩放
        # 1. 取前 3 个通道
        x_small = z[:, :3, :, :]
        # 2. 上采样 8 倍 -> (B, 3, 256, 256)
        x_recon = F.interpolate(x_small, scale_factor=8, mode='bilinear', align_corners=False)
        return x_recon

def main():
    # 检查可用设备 (优先使用 GPU/MPS)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Running training demo on device: {device}")
    
    # 初始化 Pipeline
    pipeline = StableDiffusionPipeline(device=device)

    # 确保 temp 目录存在
    os.makedirs("temp", exist_ok=True)
    
    # CRITICAL FIX: 使用 MockVAE 替换未训练的随机 VAE
    # 因为未训练的 VAE 只是随机投影，无法重建图像，会导致训练目标本身就是噪声。
    print("Replacing random VAE with MockVAE (Simple Downsample/Upsample) for demonstration...")
    pipeline.vae = MockVAE(device=device)
    
    # ---------------------------------------------------------
    # 1. 数据准备 (Data Preparation)
    # ---------------------------------------------------------
    target_img = create_target_image()
    target_img.save("temp/target_image.png")
    print(f"Created target image: temp/target_image.png (Size: {target_img.size})")
    
    # 图像预处理: (H, W, C) -> (C, H, W), 归一化到 [-1, 1]
    img_array = np.array(target_img)
    img_tensor = torch.from_numpy(img_array).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # Shape: (1, 3, 256, 256)
    
    # ---------------------------------------------------------
    # 2. VAE 编码 (VAE Encoding)
    # ---------------------------------------------------------
    print("Encoding target image to latent space...")
    with torch.no_grad():
        # VAE 将 256x256x3 压缩为 16x16x4 (压缩率 1/8 * 1/8 * 4/3 ? No, spatial 1/8)
        # 实际 VAE downsample factor 为 8 或 16，这里取决于 VAE 配置
        # 我们的 VAE 配置是 4 层 stride=2，所以是 16 倍下采样 (256/16 = 16)
        x_0 = pipeline.vae.encode(img_tensor)
        
    print(f"Latent shape: {x_0.shape}") # 预期: (1, 4, 16, 16)
        
    # 测试 VAE 重建能力 (Upper Bound)
    # 如果重建图都不清晰，说明 VAE 瓶颈，而不是 UNet 的问题
    print("Testing VAE reconstruction quality...")
    with torch.no_grad():
        reconstructed = pipeline.vae.decode(x_0)
        reconstructed_img = pipeline._tensor_to_pil(reconstructed)
        reconstructed_img.save("temp/target_reconstructed_vae.png")
    print("Saved VAE reconstruction: temp/target_reconstructed_vae.png")
    
    # ---------------------------------------------------------
    # 3. 文本编码 (Text Encoding)
    # ---------------------------------------------------------
    prompt = "a red circle"
    # 将文本转换为语义向量，用于 Conditioning
    text_emb = pipeline.text_encoder.encode([prompt])
    print(f"Text embedding shape: {text_emb.shape}") # 预期: (1, 77, 768)
    
    # ---------------------------------------------------------
    # 4. 训练前生成测试 (Zero-shot Generation)
    # ---------------------------------------------------------
    print("\nGenerating image BEFORE training (Random Weights)...")
    pipeline.unet.eval()
    with torch.no_grad():
        # 此时 UNet 只是随机猜测噪声，生成的应该是纯噪声
        # 我们使用 256x256 尺寸，这样 Latent 大小为 32x32，与训练时一致 (256/8 = 32)
        # 注意：对于简单的 DDPM 采样器，推理步数必须与训练步数一致 (1000) 才能保证时间步嵌入 (Time Embedding) 的对齐
        img_before = pipeline.generate(prompt, num_inference_steps=1000, image_size=(256, 256))
    img_before.save("temp/output_before_train.png")
    print("Saved: temp/output_before_train.png")
    
    # ---------------------------------------------------------
    # 5. 训练循环 (Training Loop)
    # ---------------------------------------------------------
    pipeline.unet.train()
    # 使用较大的学习率 (1e-3) 以便快速过拟合单张图片
    optimizer = optim.AdamW(pipeline.unet.parameters(), lr=1e-3) 
    # 添加学习率调度器：Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
    
    num_epochs = 10000 # 训练轮数
    
    print("\n" + "=" * 50)
    print("Starting training (Overfitting on single image)...")
    print("=" * 50)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        # 执行一步参数更新
        loss = train_step(pipeline, optimizer, x_0, text_emb)
        scheduler.step() # 更新学习率
        loss_history.append(loss)
        
        # 打印进度
        if (epoch + 1) % 500 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {epoch+1}/{num_epochs} | Loss: {loss:.6f} | LR: {lr:.2e}")
        
        # 定期保存生成结果，观察模型进化过程
        if (epoch + 1) % 2000 == 0:
            pipeline.unet.eval()
            with torch.no_grad():
                img_mid = pipeline.generate(prompt, num_inference_steps=1000, image_size=(256, 256))
            filename = f"temp/output_epoch_{epoch+1}.png"
            img_mid.save(filename)
            print(f"Saved intermediate result: {filename}")
            pipeline.unet.train()
            
    print("=" * 50)
    
    # ---------------------------------------------------------
    # 6. 训练后生成测试 (Final Generation)
    # ---------------------------------------------------------
    print("\nGenerating image AFTER training...")
    pipeline.unet.eval()
    with torch.no_grad():
        img_after = pipeline.generate(prompt, num_inference_steps=1000, image_size=(256, 256))
    img_after.save("temp/output_after_train.png")
    print("Saved: temp/output_after_train.png")
    
    # 结果分析
    print("\nTraining Demo Finished!")
    print(f"Initial Loss: {loss_history[0]:.6f}")
    print(f"Final Loss:   {loss_history[-1]:.6f}")
    
    if loss_history[-1] < loss_history[0] * 0.5:
        print("\nSUCCESS: Loss decreased significantly.")
        print("模型已经学会了如何去除噪声并恢复出红圈。")
    else:
        print("\nNote: Loss 下降不明显，可能需要更多训练或调整参数。")

if __name__ == "__main__":
    main()

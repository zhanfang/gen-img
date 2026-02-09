# UNet 模型

## 概述

UNet 是 Stable Diffusion 的核心“引擎”。它的任务是在扩散过程中**预测噪声**。

在每一步去噪过程中，UNet 接收当前的噪声图像（潜在表示）、当前的时间步长以及文本条件，然后输出该时间步长下图像中包含的噪声。

## 架构特点

UNet 得名于其“U”形的网络结构，主要包含三个部分：

1.  **下采样路径 (Downsampling / Encoder Path)**
2.  **瓶颈层 (Bottleneck / Middle Block)**
3.  **上采样路径 (Upsampling / Decoder Path)**

### 1. 下采样路径 (`DownBlock`)

*   **功能**: 逐步压缩图像的空间分辨率，同时增加特征通道数。这有助于模型提取高层次的语义特征。
*   **组成**: 卷积层 -> GroupNorm -> SiLU 激活 -> 下采样 (Conv2d with stride=2)。

### 2. 瓶颈层 (`BottleneckBlock`)

*   **功能**: 在最低分辨率下处理特征。这里包含了图像最抽象的全局信息。
*   **组成**: 卷积层 -> GroupNorm -> SiLU 激活。

### 3. 上采样路径 (`UpBlock`)

*   **功能**: 逐步恢复图像的空间分辨率，同时减少特征通道数。
*   **关键机制 - 跳跃连接 (Skip Connections)**:
    *   UNet 的一个关键特性是将下采样路径的特征图与对应的上采样路径的特征图进行**拼接 (Concatenation)**。
    *   这使得模型在恢复细节时，既能利用高层语义信息（来自瓶颈层），又能利用低层细节信息（来自跳跃连接）。

## 关键输入与调节 (Conditioning)

UNet 不仅仅处理图像，它还需要根据外部条件来控制生成过程：

### 1. 时间步长嵌入 (Time Embedding)

*   **问题**: 扩散过程是多步的（例如 1000 步）。模型需要知道当前处于哪一步（是刚开始全是噪声，还是快结束了只需微调？）。
*   **解决**: 将标量时间步长 $t$ 转换为向量嵌入。
*   **实现**: 使用正弦位置编码 (Sinusoidal Positional Embeddings)，类似于 Transformer 中的位置编码，然后通过 MLP 投影到特征维度。
*   **作用**: 时间嵌入被注入到每个卷积块中（通常通过相加），告诉模型当前的噪声水平。

### 2. 文本调节 (Text Conditioning)

*   **问题**: 如何让模型根据文字生成特定的图像？
*   **解决**: 使用交叉注意力机制 (Cross-Attention)。
*   **实现**: 
    *   文本被 CLIP 编码为向量序列。
    *   在 UNet 的特定层（通常是 Attention 模块），图像特征作为 Query，文本特征作为 Key 和 Value。
    *   **注**: 本项目的简化实现中，我们使用了简单的投影和相加 (`text_projection`) 来模拟这一过程，以便于学习和理解核心流程。在完整的 SD 模型中，这里是复杂的 Transformer Attention 层。

## 代码实现

在 `stable_diffusion_core/models/unet.py` 中：

```python
class UNet(nn.Module):
    def __init__(self, ...):
        # 初始化层
        self.time_embedding = TimeEmbedding(...)
        self.downs = nn.ModuleList(...)
        self.ups = nn.ModuleList(...)
        ...

    def forward(self, x, t, text_emb=None):
        # 1. 嵌入时间
        t_emb = self.time_embedding(t)
        
        # 2. 融合文本条件
        # (简化版：投影并相加)
        
        # 3. 下采样 (保存跳跃连接)
        for down in self.downs:
            skip, x = down(x, t_emb)
            skip_connections.append(skip)
            
        # 4. 瓶颈处理
        x = self.bottleneck(x, t_emb)
        
        # 5. 上采样 (拼接跳跃连接)
        for up in self.ups:
            x = up(x, skip_connections.pop(), t_emb)
            
        return output
```

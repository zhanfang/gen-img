# 文本编码器 (Text Encoder)

## 概述

为了让 Stable Diffusion 理解用户的输入（例如 "a cute cat"），我们需要将文本转换为机器可以理解的数学表示（向量）。这正是文本编码器的作用。

Stable Diffusion 并没有从头开始训练一个文本理解模型，而是利用了预训练的 **CLIP (Contrastive Language-Image Pre-Training)** 模型。

## CLIP 模型

CLIP 是 OpenAI 开发的一个多模态模型，它在一个巨大的图像-文本对数据集上进行了训练。

*   **特点**: CLIP 的文本编码器可以将文本映射到一个与图像共享的潜在空间。这意味着语义相关的文本和图像在这个空间中的距离会很近。
*   **架构**: CLIP 的文本编码器通常是一个 Transformer 模型（类似于 BERT 或 GPT）。

## 处理流程

1.  **分词 (Tokenization)**
    *   **输入**: 原始文本字符串，如 "Hello world"。
    *   **过程**: 使用 tokenizer 将文本分解为 token ID 序列。例如，"Hello world" 可能变成 `[4911, 1922]`。
    *   **特殊 Token**: 
        *   `[SOS]` (Start of Sequence): 序列开始。
        *   `[EOS]` (End of Sequence): 序列结束。
        *   `[PAD]`: 填充，确保所有输入长度一致（CLIP 默认为 77）。

2.  **编码 (Encoding)**
    *   **输入**: Token ID 序列。
    *   **过程**: 通过 Transformer 模型处理。
    *   **输出**: 文本嵌入 (Embeddings)。
        *   形状通常为 `(Batch, Sequence_Length, Hidden_Dim)`。
        *   例如：`(1, 77, 768)`。

## 在 Stable Diffusion 中的作用

生成的文本嵌入被传递给 **UNet**。UNet 通过**交叉注意力 (Cross-Attention)** 机制，利用这些嵌入来指导图像生成过程。

*   当你说 "画一只猫" 时，"猫" 的向量表示会“告诉” UNet 在图像的相应区域增强类似猫的特征。

## 代码实现

在 `stable_diffusion_core/models/text_encoder.py` 中，我们封装了 `transformers` 库中的 CLIP 模型：

```python
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder:
    def __init__(self, ...):
        # 加载 tokenizer 和模型
        self.tokenizer = CLIPTokenizer.from_pretrained(...)
        self.model = CLIPTextModel(...)
        
    def encode(self, texts):
        # 1. Tokenize
        inputs = self.tokenizer(texts, ...)
        
        # 2. Forward pass 获取 hidden states
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
```

> **注意**: 在本项目的演示代码中，为了方便无网络环境或避免下载大模型，我们默认使用了随机初始化的配置。在实际生产环境中，必须加载预训练的权重才能生成有意义的图像。

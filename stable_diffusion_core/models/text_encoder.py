from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import torch

class TextEncoder:
    """
    文本编码器，用于将文本提示转换为嵌入向量。
    
    通常使用预训练的 CLIP 模型（Contrastive Language-Image Pre-Training）。
    它将文本转换为 UNet 可以理解的向量表示，从而实现文本条件生成。
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        """
        初始化文本编码器。
        
        Args:
            model_name (str): 预训练 CLIP 模型的名称。
            device (str): 运行设备 ('cuda' 或 'cpu')。
        """
        self.model_name = model_name
        self.device = device
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Failed to load CLIPTokenizer: {e}")
            self.tokenizer = None
            
        # 使用随机初始化以避免兼容性问题和大型下载 (用于学习目的)
        # 在实际应用中，应该加载预训练权重：CLIPTextModel.from_pretrained(model_name)
        config = CLIPTextConfig()
        self.model = CLIPTextModel(config).to(device)
    
    def encode(self, texts, max_length=77):
        """
        将文本提示编码为嵌入。
        
        Args:
            texts (list): 文本提示字符串列表。
            max_length (int): 最大 token 长度 (CLIP 默认为 77)。
            
        Returns:
            torch.Tensor: 文本嵌入张量，形状为 (batch_size, seq_len, hidden_size)。
        """
        if self.tokenizer is None:
            # 如果 tokenizer 加载失败，返回随机嵌入 (仅用于测试/演示)
            batch_size = len(texts)
            return torch.randn(batch_size, max_length, self.model.config.hidden_size).to(self.device)
            
        # 对文本进行 Tokenize
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 获取嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        return embeddings
    
    def get_text_embedding_dim(self):
        """
        获取文本嵌入的维度。
        
        Returns:
            int: 隐藏层大小。
        """
        return self.model.config.hidden_size

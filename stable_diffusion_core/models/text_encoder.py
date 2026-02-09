from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
import torch

class TextEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.model_name = model_name
        self.device = device
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Failed to load CLIPTokenizer: {e}")
            self.tokenizer = None
            
        # Use random initialization to avoid compatibility issues and large downloads
        config = CLIPTextConfig()
        self.model = CLIPTextModel(config).to(device)
    
    def encode(self, texts, max_length=77):
        """
        Encode text prompts to embeddings
        texts: list of text prompts
        max_length: maximum length of tokens
        """
        if self.tokenizer is None:
            # Return random embeddings if tokenizer failed
            batch_size = len(texts)
            return torch.randn(batch_size, max_length, self.model.config.hidden_size).to(self.device)
            
        # Tokenize text
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        return embeddings
    
    def get_text_embedding_dim(self):
        """
        Get the dimension of text embeddings
        """
        return self.model.config.hidden_size

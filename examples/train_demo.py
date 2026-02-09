import torch
import torch.optim as optim
from PIL import Image, ImageDraw
import numpy as np
from stable_diffusion_core.pipelines.stable_diffusion_pipeline import StableDiffusionPipeline
import os

def create_target_image():
    """Create a simple 256x256 RGB image with a red circle on white background"""
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    # Draw a red circle
    draw.ellipse([64, 64, 192, 192], fill='red', outline='red')
    return img

def train_step(pipeline, optimizer, x_0, text_emb):
    optimizer.zero_grad()
    # Calculate loss using the diffusion process utility
    loss = pipeline.diffusion.get_loss(pipeline.unet, x_0, text_emb=text_emb)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    # Use CPU for stability in this demo if GPU is not explicitly strong, 
    # but try CUDA/MPS if available.
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Running training demo on device: {device}")
    
    # Initialize pipeline
    pipeline = StableDiffusionPipeline(device=device)
    
    # 1. Create and Process Data
    target_img = create_target_image()
    target_img.save("target_image.png")
    print("Created target image: target_image.png")
    
    # Preprocess image to tensor [-1, 1]
    # (H, W, C) -> (C, H, W)
    img_array = np.array(target_img)
    img_tensor = torch.from_numpy(img_array).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, 64, 64)
    
    # Encode to latent space using VAE
    print("Encoding target image to latent space...")
    with torch.no_grad():
        x_0 = pipeline.vae.encode(img_tensor)
    
    # Get text embedding for a fixed prompt
    prompt = "a red circle"
    text_emb = pipeline.text_encoder.encode([prompt])
    
    # 2. Generate BEFORE training (Random Weights)
    print("\nGenerating image BEFORE training...")
    pipeline.unet.eval()
    with torch.no_grad():
        # Using fewer steps for speed in demo
        img_before = pipeline.generate(prompt, num_inference_steps=20, image_size=(128, 128))
    img_before.save("output_before_train.png")
    print("Saved: output_before_train.png")
    
    # 3. Training Loop
    pipeline.unet.train()
    optimizer = optim.AdamW(pipeline.unet.parameters(), lr=1e-3) # Higher LR for fast overfitting
    num_epochs = 100
    
    print("\n" + "=" * 50)
    print("Starting training (Overfitting on single image)...")
    print("=" * 50)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        loss = train_step(pipeline, optimizer, x_0, text_emb)
        loss_history.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Step {epoch+1}/{num_epochs} | Loss: {loss:.6f}")
            
    print("=" * 50)
    
    # 4. Generate AFTER training
    print("\nGenerating image AFTER training...")
    pipeline.unet.eval()
    with torch.no_grad():
        # We need to ensure we use the same text embedding (which we do inside generate)
        img_after = pipeline.generate(prompt, num_inference_steps=20, image_size=(128, 128))
    img_after.save("output_after_train.png")
    print("Saved: output_after_train.png")
    
    print("\nTraining Demo Finished!")
    print(f"Initial Loss: {loss_history[0]:.6f}")
    print(f"Final Loss:   {loss_history[-1]:.6f}")
    
    if loss_history[-1] < loss_history[0]:
        print("\nSUCCESS: Loss decreased significantly.")
        print("The 'output_after_train.png' should resemble 'target_image.png' much more than 'output_before_train.png'.")
    else:
        print("\nNote: Loss did not decrease as expected. Check hyperparameters.")

if __name__ == "__main__":
    main()

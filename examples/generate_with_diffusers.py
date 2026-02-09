import torch
from diffusers import StableDiffusionPipeline
import sys

def main():
    """
    This script uses the official Hugging Face 'diffusers' library to generate 
    images using pre-trained Stable Diffusion weights.
    
    This demonstrates what the output SHOULD look like if our custom model 
    was fully trained or loaded with these weights.
    """
    if len(sys.argv) < 2:
        print("Usage: python generate_with_diffusers.py <prompt>")
        print("Example: python generate_with_diffusers.py 'a cat sitting on a bench'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    print(f"Generating image for prompt: {prompt}")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading pre-trained model: {model_id}...")
    print("Note: This requires an internet connection to download model weights (approx. 4GB).")
    
    try:
        # Use float16 for faster inference if on GPU (MPS for Mac or CUDA)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype
        )
        pipe = pipe.to(device)
        
        # Enable attention slicing for lower memory usage
        pipe.enable_attention_slicing()
        
        print(f"Model loaded on {device}. Starting inference...")
        
        image = pipe(prompt).images[0]
        
        output_path = f"output_pretrained_{prompt[:20].replace(' ', '_')}.png"
        image.save(output_path)
        print(f"Image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure you have internet access and 'diffusers' installed.")
        print("pip install diffusers transformers accelerate")

if __name__ == "__main__":
    main()

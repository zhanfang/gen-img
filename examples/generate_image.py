import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_diffusion_core.pipelines.stable_diffusion_pipeline import StableDiffusionPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_image.py <prompt>")
        print("Example: python generate_image.py 'a cat sitting on a bench'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    print(f"Generating image for prompt: {prompt}")
    
    # Initialize pipeline
    pipeline = StableDiffusionPipeline()
    
    # Generate image
    image = pipeline.generate(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        image_size=(512, 512)
    )
    
    # Save image
    output_path = f"output_{prompt[:20].replace(' ', '_')}.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    # Display image
    # image.show()

if __name__ == "__main__":
    main()

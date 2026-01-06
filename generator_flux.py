"""
Minimal FLUX Image Generation with GGUF Quantization
Simple class wrapper around the working zz.py code
"""

import torch
import os
from PIL import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig


class FluxGenerator:
    """Simple FLUX generator using GGUF quantization - minimal working version"""
    
    def __init__(self):
        """Initialize the FLUX pipeline with GGUF quantization"""
        print("Loading FLUX model with GGUF quantization...")
        
        # Paths from zz.py
        ckpt_path = os.path.join("models", "flux1-dev-Q4_0.gguf")
        path_lora = os.path.join('models', 'FluxDFaeTasticDetails.safetensors')
        path_lora = os.path.join("models", "aidmaImageUpgrader-FLUX-V0.1.safetensors")
        # path_lora = os.path.join("models", "lineart_flux.safetensors")
        # Load transformer with GGUF quantization (from zz.py lines 9-13)
        self.transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        
        # Load pipeline (from zz.py lines 15-19)
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=self.transformer,
            torch_dtype=torch.bfloat16,
        )
        
        # Load LoRA weights (from zz.py lines 20-21)
        if os.path.exists(path_lora):
            self.pipe.load_lora_weights(path_lora)
        
        # Enable CPU offload (from zz.py line 22)
        self.pipe.enable_model_cpu_offload()
        
        print("✅ Generator ready")
    
    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        seed: int = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of desired image
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of denoising steps
            seed: Random seed for reproducibility (None for random)
        
        Returns:
            PIL Image object
        """
        print(f"🎨 Generating: '{prompt[:50]}...'")
        
        # Handle seed: if None, use None (random), otherwise create generator with seed
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        
        # Generate image (from zz.py lines 25-31)
        images = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=generator
        ).images[0]
        
        print("✅ Generation complete")
        return images
    
    def save_image(self, image: Image.Image, filename: str) -> str:
        """
        Save generated image to disk.
        
        Args:
            image: PIL Image to save
            filename: Name of the file (with or without extension)
        
        Returns:
            Full path to saved image
        """
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
        
        image.save(os.path.join("outputs", filename))
        print(f"💾 Saved to: {filename}")
        return filename


def main():
    """Simple test"""
    generator = FluxGenerator()
    
    prompt = "a moonim dressed as a knight, riding a horse towards a medieval castle"
    image = generator.generate_image(prompt, seed=42)
    generator.save_image(image, "sd")


if __name__ == "__main__":
    main()

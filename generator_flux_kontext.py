"""
FLUX Kontext Image Generation with GGUF Quantization
Image editing using reference image and prompt
"""

import torch
import os
from PIL import Image
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig


class FluxKontextGenerator:
    """FLUX Kontext generator using GGUF quantization for image editing"""
    
    def __init__(self):
        """Initialize the FLUX Kontext pipeline with GGUF quantization"""
        print("Loading FLUX Kontext model with GGUF quantization...")
        
        # Paths
        ckpt_path = os.path.join("models", "flux1-kontext-dev-Q4_0.gguf")
        # path_lora = os.path.join("models", "aidmaImageUpgrader-FLUX-V0.1.safetensors")
        
        # Load transformer with GGUF quantization
        self.transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        
        # Load FLUX Kontext pipeline
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            transformer=self.transformer,
            torch_dtype=torch.bfloat16,
        )
        
        # Load LoRA weights
        # if os.path.exists(path_lora):
        #     self.pipe.load_lora_weights(path_lora)
        
        # Enable CPU offload
        self.pipe.enable_model_cpu_offload()
        self.pipe.transformer.config.in_channels = 64
        
        print("✅ Generator ready")
    
    def generate_image(
        self,
        prompt: str,
        reference_image: Image.Image,
        steps: int = 20,
        seed: int = None
    ) -> Image.Image:
        """
        Generate/Edit an image from a reference image and text prompt.
        
        Args:
            prompt: Text description of desired image changes
            reference_image: Reference image to edit
            steps: Number of denoising steps
            seed: Random seed for reproducibility (None for random)
        
        Returns:
            PIL Image object
        """
        print(f"🎨 Generating: '{prompt[:50]}...'")
        
        # Ensure reference image is RGB
        if reference_image.mode != "RGB":
            reference_image = reference_image.convert("RGB")
        
        # FLUX Kontext works best with preferred resolutions
        # Resize to 1024x1024 which is in the preferred resolutions list
        # This ensures proper latent packing
        if reference_image.size != (1024, 1024):
            reference_image = reference_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Handle seed: if None, use None (random), otherwise create generator with seed
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        
        # Generate image with reference
        # FLUX Kontext uses 'image' parameter for reference image
        # Image is already 1024x1024 (preferred resolution), so _auto_resize will keep it
        images = self.pipe(
            prompt=prompt,
            image=reference_image,
            num_inference_steps=steps,
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
    generator = FluxKontextGenerator()
    
    # Load a test reference image
    reference = Image.new("RGB", (1024, 1024), color="white")
    prompt = "add a beautiful sunset in the background"
    image = generator.generate_image(prompt, reference, seed=42)
    generator.save_image(image, "test_kontext")


if __name__ == "__main__":
    main()


"""
Gradio Web Interface for FLUX Image Generation

This provides an easy-to-use web interface for generating images
using FLUX with GGUF quantization.

Run with: python gradio_flux_app.py
"""

import gradio as gr
from generator_flux import FluxGenerator
from PIL import Image
import os
import random
from datetime import datetime


class GradioFluxApp:
    """Gradio interface wrapper for the FLUX generator."""
    
    def __init__(self):
        """Initialize the generator and Gradio interface."""
        print("🚀 Starting Gradio interface...")
        self.generator = FluxGenerator()
        self.interface = self._create_interface()
    
    def generate_wrapper(
        self,
        prompt: str,
        aspect_ratio: str,
        steps: int,
        seed: int,
        use_seed: bool
    ) -> tuple:
        """
        Wrapper function for Gradio that handles the generation.
        
        Args:
            prompt: Text prompt for image generation
            aspect_ratio: Aspect ratio string (e.g., "1:1", "16:9")
            steps: Number of inference steps
            seed: Random seed
            use_seed: Whether to use the seed or randomize
        
        Returns:
            Tuple of (Generated PIL Image, seed text)
        """
        # Validate prompt
        if not prompt or prompt.strip() == "":
            return None, "Seed: -"
        
        # Map aspect ratios to dimensions (all multiples of 64 for FLUX)
        ratio_map = {
            "1:1": (1024, 1024),
            "3:4": (768, 1024),   # Portrait
            "4:3": (1024, 768),   # Landscape
            "16:9": (1024, 576),  # Landscape
            "9:16": (576, 1024)   # Portrait
        }
        
        width, height = ratio_map.get(aspect_ratio, (1024, 1024))
        
        # Handle seed: use seed only if checkbox is checked and seed is valid
        # If no seed is used, generate a random one to display
        actual_seed = None
        if use_seed and seed is not None:
            try:
                actual_seed = int(seed)
            except (ValueError, TypeError):
                actual_seed = None
        
        # If no seed was provided, generate a random one for display
        if actual_seed is None:
            actual_seed = random.randint(0, 2147483647)
        
        # Generate image
        try:
            image = self.generator.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                seed=actual_seed
            )
            
            # Auto-save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gradio_flux_{timestamp}"
            self.generator.save_image(image, filename)
            
            # Return image and seed text
            seed_text = f"Seed: {actual_seed}"
            return image, seed_text
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            raise gr.Error(f"Generation failed: {str(e)}")
    
    def _create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface with all components.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="FLUX Image Generator") as interface:
            gr.Markdown(
                """
                # 🎨 FLUX Image Generator
                
                Generate high-quality images from text descriptions using FLUX with GGUF quantization.
                Enter your prompt and adjust parameters below.
                """
            )
            
            with gr.Row():
                # Left column: Inputs
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="a moonim dressed as a knight, riding a horse towards a medieval castle",
                        lines=3,
                        max_lines=5
                    )
                    
                    # Aspect ratio buttons
                    gr.Markdown("**Aspect Ratio:**")
                    aspect_ratio = gr.Radio(
                        choices=["1:1", "3:4", "4:3", "16:9", "9:16"],
                        value="1:1",
                        label="",
                        show_label=False,
                        info="1:1 (1024×1024) | 3:4 (768×1024) | 4:3 (1024×768) | 16:9 (1024×576) | 9:16 (576×1024)"
                    )
                    
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        steps_slider = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=20,
                            step=1,
                            label="Steps",
                            info="More steps = better quality but slower"
                        )
                        
                        with gr.Row():
                            use_seed_checkbox = gr.Checkbox(
                                label="Use fixed seed",
                                value=False,
                                info="Enable for reproducible results"
                            )
                            
                            seed_input = gr.Number(
                                label="Seed",
                                value=42,
                                precision=0,
                                minimum=0,
                                maximum=2147483647
                            )
                    
                    generate_btn = gr.Button(
                        "🎨 Generate Image",
                        variant="primary",
                        size="lg"
                    )
                
                # Right column: Output
                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Generated Image",
                        type="pil",
                        height=512
                    )
                    
                    seed_display = gr.Textbox(
                        label="",
                        value="Seed: -",
                        interactive=False,
                        show_label=False
                    )
                    
                    gr.Markdown(
                        """
                        ### 💡 Tips:
                        - Be specific and descriptive in your prompts
                        - Use quality keywords: "detailed", "4k", "masterpiece"
                        - Default: 20 steps (good balance of quality and speed)
                        - More steps = better quality but slower
                        - Images auto-save to current directory
                        """
                    )
            
            # Examples section
            gr.Markdown("### 📸 Example Prompts")
            gr.Examples(
                examples=[
                    [
                        "a moonim dressed as a knight, riding a horse towards a medieval castle",
                        "1:1",
                        20
                    ],
                    [
                        "A serene mountain landscape at sunset, highly detailed, 4k",
                        "16:9",
                        20
                    ],
                    [
                        "Cyberpunk portrait of a woman with neon hair, futuristic city background, digital art",
                        "9:16",
                        20
                    ],
                    [
                        "Abstract geometric patterns, vibrant colors, modern art style",
                        "1:1",
                        20
                    ],
                    [
                        "A cute robot reading a book in a cozy library, warm lighting, digital illustration",
                        "4:3",
                        20
                    ],
                    [
                        "Underwater coral reef, tropical fish, sunlight rays, highly detailed, nature photography",
                        "16:9",
                        20
                    ]
                ],
                inputs=[prompt_input, aspect_ratio, steps_slider],
                outputs=[output_image, seed_display],
                fn=lambda p, ar, s: self.generate_wrapper(p, ar, s, 42, False),
                cache_examples=False
            )
            
            # Connect the generate button
            generate_btn.click(
                fn=self.generate_wrapper,
                inputs=[
                    prompt_input,
                    aspect_ratio,
                    steps_slider,
                    seed_input,
                    use_seed_checkbox
                ],
                outputs=[output_image, seed_display],
                queue=False
            )
        
        return interface
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public share link
            server_port: Port to run the server on
        """
        print("\n" + "="*60)
        print("🌐 Launching web interface...")
        print(f"📍 Local URL: http://localhost:{server_port}")
        if share:
            print("🔗 Public share link will be generated...")
        print("="*60 + "\n")
        
        self.interface.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0"  # Allow external connections
        )


def main():
    """Launch the Gradio app."""
    app = GradioFluxApp()
    app.launch(share=False)  # Set share=True to create public link


if __name__ == "__main__":
    main()


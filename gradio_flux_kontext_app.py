"""
Gradio Web Interface for FLUX Kontext Image Editing

This provides an easy-to-use web interface for editing images
using FLUX Kontext with GGUF quantization and reference images.

Run with: python gradio_flux_kontext_app.py
"""

import gradio as gr
from generator_flux_kontext import FluxKontextGenerator
from PIL import Image
import os
import random
from datetime import datetime


class GradioFluxKontextApp:
    """Gradio interface wrapper for the FLUX Kontext generator."""
    
    def __init__(self):
        """Initialize the generator and Gradio interface."""
        print("🚀 Starting Gradio Kontext interface...")
        self.generator = FluxKontextGenerator()
        self.interface = self._create_interface()
    
    def generate_wrapper(
        self,
        reference_image: Image.Image,
        prompt: str,
        steps: int,
        seed: int,
        use_seed: bool
    ) -> tuple:
        """
        Wrapper function for Gradio that handles the generation.
        
        Args:
            reference_image: Reference image to edit
            prompt: Text prompt for image editing
            steps: Number of inference steps
            seed: Random seed
            use_seed: Whether to use the seed or randomize
        
        Returns:
            Tuple of (Generated PIL Image, seed text)
        """
        # Validate inputs
        if reference_image is None:
            return None, "Seed: - (Please upload a reference image)"
        
        if not prompt or prompt.strip() == "":
            return None, "Seed: - (Please enter a prompt)"
        
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
                reference_image=reference_image,
                steps=steps,
                seed=actual_seed
            )
            
            # Auto-save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gradio_flux_kontext_{timestamp}"
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
        with gr.Blocks(title="FLUX Kontext Image Editor") as interface:
            gr.Markdown(
                """
                # 🎨 FLUX Kontext Image Editor
                
                Edit images using FLUX Kontext with GGUF quantization.
                Upload a reference image and describe how you want to edit it.
                Images are generated at 1024×1024 (1:1 aspect ratio).
                """
            )
            
            with gr.Row():
                # Left column: Inputs
                with gr.Column(scale=1):
                    reference_image_input = gr.Image(
                        label="Reference Image",
                        type="pil",
                        height=400
                    )
                    
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="add a beautiful sunset in the background, make it more vibrant",
                        lines=3,
                        max_lines=5,
                        info="Describe how you want to edit the image"
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
                        "🎨 Edit Image",
                        variant="primary",
                        size="lg"
                    )
                
                # Right column: Output
                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Edited Image",
                        type="pil",
                        height=400
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
                        - Upload a reference image (will be resized to 1024×1024)
                        - Be specific about what you want to change or add
                        - Use descriptive prompts: "add", "change", "make it", "transform"
                        - Default: 20 steps (good balance of quality and speed)
                        - More steps = better quality but slower
                        - Images auto-save to outputs/ directory
                        """
                    )
            
            # Examples section
            gr.Markdown("### 📸 Example Prompts")
            gr.Examples(
                examples=[
                    [
                        "add a beautiful sunset in the background with warm colors",
                        20
                    ],
                    [
                        "make the image more vibrant and colorful, enhance the lighting",
                        20
                    ],
                    [
                        "transform into a cyberpunk style with neon lights",
                        25
                    ],
                    [
                        "add dramatic clouds and change the mood to more mysterious",
                        20
                    ],
                    [
                        "make it look like a watercolor painting",
                        20
                    ]
                ],
                inputs=[prompt_input, steps_slider],
                outputs=[output_image, seed_display],
                fn=lambda p, s: self.generate_wrapper(None, p, s, 42, False),
                cache_examples=False
            )
            
            # Connect the generate button
            generate_btn.click(
                fn=self.generate_wrapper,
                inputs=[
                    reference_image_input,
                    prompt_input,
                    steps_slider,
                    seed_input,
                    use_seed_checkbox
                ],
                outputs=[output_image, seed_display],
                queue=False
            )
        
        return interface
    
    def launch(self, share: bool = False, server_port: int = 7861):
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public share link
            server_port: Port to run the server on (default 7861 to avoid conflict)
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
    app = GradioFluxKontextApp()
    app.launch(share=False)  # Set share=True to create public link


if __name__ == "__main__":
    main()


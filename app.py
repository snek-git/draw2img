import gradio as gr
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL, StableDiffusionXLPipeline
import torch

# Function to generate an image
def generate_image(prompt, negative_prompt, cfg_scale, num_inference_steps, height, width):
    if torch.cuda.is_available():  # Check for CUDA GPU
        with torch.autocast("cuda"):
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width
            ).images[0]
    else:  # Use MPS by default
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width
        ).images[0]
    return image

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",  
    torch_dtype=torch.float16
)
pipe = StableDiffusionXLPipeline.from_single_file(
    "./models/animagine-xl-3.0.safetensors",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Determine device for optimization
if torch.cuda.is_available():
    pipe.to("cuda")
else:
    pipe.to("mps")  # For Apple M1/M2 chips

# Gradio interface - Set default resolution here
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(lines=1, placeholder="Enter image description"),
        gr.Textbox(lines=1, placeholder="Enter negative prompt"),
        gr.Slider(0, 10, value=7, step=0.5, label="CFG Scale"),
        gr.Slider(0, 50, value=28, step=1, label="Inference Steps"),
        gr.Number(value=1216, label="Height"), 
        gr.Number(value=832, label="Width")  
    ],
    outputs="image"
)
demo.launch()

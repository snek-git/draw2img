import gradio as gr
from diffusers import UNet2DConditionModel, AutoPipelineForImage2Image, LCMScheduler
import torch

# Load the LCM model
unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Check for GPU and move model appropriately
# if torch.cuda.is_available():
pipe.to("cuda")
# else:
#     pipe.to("mps")  # For Apple M1/M2 chips

# Function to generate an image
def generate_image(prompt, init_image):
    # if torch.cuda.is_available():
    with torch.autocast("cuda"):
        image = pipe(
            prompt = prompt,
            image = init_image,
            guidance_scale=7,
            num_inference_steps=4,
            strength = 0.4,
            height=1024,
            width=1024
        ).images[0]
    # else:
    #     image = pipe(
    #         prompt = prompt,
    #         image = init_image,
    #         guidance_scale=7,
    #         num_inference_steps=4,
    #         strength = 0.4,
    #         height=1024,
    #         width=1024
    #     ).images[0]
    return image

# Gradio interface (same as before)
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(lines=1, placeholder="Describe your image"),
        gr.Image()
    ],
    outputs="image"
)

demo.launch()
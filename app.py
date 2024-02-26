# import gradio as gr
# from diffusers import UNet2DConditionModel, AutoPipelineForImage2Image, LCMScheduler, DiffusionPipeline
# import torch

# # Load the LCM model
# unet = UNet2DConditionModel.from_pretrained(
#     "latent-consistency/lcm-sdxl",
#     torch_dtype=torch.float16,
#     variant="fp16",
# )

# # pipe = AutoPipelineForImage2Image.from_pretrained(
# #     "stabilityai/stable-diffusion-xl-base-1.0",
# #     unet=unet,
# #     torch_dtype=torch.float16,
# #     variant="fp16",
# # )

# pipe = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     unet=unet,
#     torch_dtype=torch.float16,
#     variant="fp16"
# )

# pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# # Check for GPU and move model appropriately
# if torch.cuda.is_available():
#     pipe.to("cuda")
# else:
#     pipe.to("mps")  # For Apple M1/M2 chips

# # Function to generate an image
# def generate_image(prompt):
#     if torch.cuda.is_available():
#         with torch.autocast("cuda"):
#             image = pipe(
#                 prompt = prompt,
#                 # image = init_image,
#                 guidance_scale=7,
#                 num_inference_steps=4,
#                 strength = 0.4,
#                 height=1024,
#                 width=1024
#             ).images[0]
#     else:
#         image = pipe(
#             prompt = prompt,
#             # image = init_image,
#             guidance_scale=7,
#             num_inference_steps=4,
#             strength = 0.4,
#             height=1024,
#             width=1024
#         ).images[0]
#     return image

# # Gradio interface (same as before)
# demo = gr.Interface(
#     fn=generate_image,
#     inputs=[
#         gr.Textbox(lines=1, placeholder="Describe your image"),
#         # gr.Image()
#     ],
#     outputs="image"
# )

# demo.launch()


import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import make_image_grid, load_image
import gradio as gr
import cv2

unet = UNet2DConditionModel.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    subfolder="unet",
    torch_dtype=torch.float16,
)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "Lykon/dreamshaper-7",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# prepare image
# prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
# generator = torch.manual_seed(0)

def generate_image(prompt,init_image):
    
    init_image = cv2.resize(init_image, dsize = (1024, 1024), interpolation=cv2.INTER_CUBIC)
    
    image = pipe(
        prompt = prompt,
        image=init_image,
        num_inference_steps=4,
        guidance_scale=2,
        strength=0.8,
    ).images[0]
    
    return image

demo = gr.Interface(
    fn=generate_image,     
    inputs=[
        gr.Textbox(lines=1, placeholder="Describe your image"),
        gr.Image()
    ],
    outputs="image"
)

demo.launch()
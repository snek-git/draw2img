from flask import Flask, request, jsonify
import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
# from diffusers.utils import make_image_grid, load_image
import gradio as gr
import cv2
import base64
import io
from PIL import Image

unet = UNet2DConditionModel.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    # "latent-consistency/lcm-sdxl",
    subfolder="unet",
    torch_dtype=torch.float16,
)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "Lykon/dreamshaper-7",
    # "stabilityai/stable-diffusion-xl-base-1.0",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)


app = Flask(__name__)


def generate_image(prompt,init_image, steps, cfg, strength):
    
    init_image = cv2.resize(init_image, dsize = (1024, 1024), interpolation=cv2.INTER_CUBIC)
    
    image = pipe(
        prompt = prompt, # "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
        image=init_image,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=strength,
    ).images[0]
    
    return image

@app.route('/generate', methods=['POST'])
def generate():
    
    prompt = request.json.get('prompt')
    init_image_data = request.json.get('init_image') 
    steps = request.json.get('steps')
    cfg = request.json.get('cfg')
    strength = request.json.get('strength')
    
    try:
        image_bytes = base64.b64decode(init_image_data)
        init_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({'error': 'Invalid base64 image data' + e.toString()})
    
    image = generate_image(prompt, init_image, steps, cfg, strength)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'image': image_data})
    

demo = gr.Interface(
    fn=generate_image,     
    inputs=[
        gr.Textbox(lines=1, placeholder="Describe your image"),
        gr.Image(),
        gr.Slider(1, 10, 1, 1, label="Steps"),
        gr.Slider(1, 10, 1, 1, label="Guidance Scale"),
        gr.Slider(0.1, 1, 0.1, 0.1, label="Strength"),
    ],
    outputs="image"
)

demo.launch()

# run flask app
# if __name__ == '__main__':
#     app.run(debug=True)
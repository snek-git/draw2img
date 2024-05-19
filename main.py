import os
import time
from sys import platform
import torch
from datetime import datetime
from PIL import Image
import gradio as gr
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from diffusers import AutoPipelineForImage2Image, LCMScheduler
import random
import io
import base64

# Define the cache directory path
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Set environment variables for cache directories
os.environ.update({
    "TRANSFORMERS_CACHE": cache_dir,
    "HF_HUB_CACHE": cache_dir,
    "HF_HOME": cache_dir
})

is_mac = platform == "darwin"

def use_fp16():
    if is_mac:
        return True
    # Check if the GPU supports TensorFloat-32 (TF32)
    # Major compute capability >= 6.0
    gpu_props = torch.cuda.get_device_properties("cuda")
    return gpu_props.major >= 6

class Timer:
    def __init__(self, task_name="timed process"):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        print(f"{self.task_name} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(f"{self.task_name} took {round(end_time - self.start_time, 2)}s")

def load_pipeline(model_id="Lykon/dreamshaper-7"):
    if not is_mac:
        torch.backends.cuda.matmul.allow_tf32 = True

    use_half_precision = use_fp16()
    lora_id = "latent-consistency/lcm-lora-sdv1-5"

    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if use_half_precision else torch.float32,
        variant="fp16" if use_half_precision else None,
        safety_checker=None
    )

    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(lora_id)
    pipeline.fuse_lora()

    device = "mps" if is_mac else "cuda"
    pipeline.to(device=device)
    return pipeline

# Global variable for the inference pipeline
pipeline = load_pipeline()

def generate_image(prompt, image, steps, cfg_scale, sketch_strength):
    generator = torch.manual_seed(random.randint(0, 100000))
    with torch.autocast("cuda"):
        return pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            strength=sketch_strength,
            generator=generator
        ).images[0]


def save_image(prompt, image, steps, cfg_scale, sketch_strength):
    generated_image = generate_image(prompt, image, steps, cfg_scale, sketch_strength)
    
    os.makedirs("images", exist_ok=True)
    
    # Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"images/img_generated{timestamp}.png"
    
    generated_image.save(filename)
    print(f"Image saved as {filename}")
    return generated_image
    

def create_interface():
    canvas_size = 512

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                steps_slider = gr.Slider(label="Steps", minimum=1, maximum=10, step=1, value=4, interactive=True)
                cfg_slider = gr.Slider(label="CFG", minimum=1, maximum=5, step=1, value=1, interactive=True)
                strength_slider = gr.Slider(label="Sketch/Prompt Strength", minimum=0.1, maximum=1, step=0.1, value=0.9, interactive=True)
            with gr.Column():
                prompt_input = gr.Textbox(label="Prompt", value="flock of birds flying over the sea, splash art", interactive=True)
        with gr.Row(equal_height=True):
            input_image = gr.Image(source="canvas", tool="color-sketch", shape=(canvas_size, canvas_size), width=canvas_size, height=canvas_size, type="pil")
            output_image = gr.Image(width=canvas_size, height=canvas_size)

            def process_image(prompt, image, steps, cfg, sketch_strength):
                if not image:
                    return Image.new("RGB", (canvas_size, canvas_size))
                return generate_image(
                    prompt=prompt,
                    image=image,
                    steps=steps,
                    cfg_scale=cfg,
                    sketch_strength=sketch_strength,
                )

            controls = [prompt_input, input_image, steps_slider, cfg_slider, strength_slider]

            for control in controls:
                control.change(fn=process_image, inputs=controls, outputs=output_image)

    return demo

app = Flask(__name__)
CORS(app) 


def convert_base64_to_image(base64_str):
    # Decode the base64 string to get the binary data
    image_data = base64.b64decode(base64_str)
    # Convert the binary data to a PIL Image object
    image = Image.open(io.BytesIO(image_data))
    # Ensure the image is in RGB format
    image = image.convert("RGB")
    return image

@app.route("/generate", methods=["POST"])
def api_generate_image():
    data = request.json
    prompt = data.get("prompt", "")
    steps = data.get("steps", 4)
    cfg_scale = data.get("cfg_scale", 1)
    strength = data.get("strength", 0.9)

    image_data = data.get("image")
    if image_data:
        image = convert_base64_to_image(image_data.split("data:image/octet-stream;base64,")[1])
        print("Image received")
    else:
        return jsonify({"error": "No image provided"}), 400

    with Timer("Image generation"):
        print("Generating image FELOFELEFL")
        generated_image = save_image(prompt, image, steps, cfg_scale, strength)

    buffered = io.BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"image": img_str})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Deploy on Gradio for public URL", default=False)
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask API")
    parser.add_argument("--gradio", action="store_true",default = False, help="Run Gradio interface")
    args = parser.parse_args()

    interface = create_interface()
    
    # Run Gradio and Flask in parallel
    from threading import Thread

    def run_gradio():
        interface.launch(share=args.share)

    def run_flask():
        app.run(port=args.port)

    if args.gradio:
        gradio_thread = Thread(target=run_gradio)
        flask_thread = Thread(target=run_flask)

        gradio_thread.start()
        flask_thread.start()

        gradio_thread.join()
        flask_thread.join()
    else:
        run_flask()


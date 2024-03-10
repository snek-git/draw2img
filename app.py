import base64
import cv2
import io
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
)
from flask import Flask, request, jsonify
from transformers import ViTFeatureExtractor, ViTForImageClassification


# Load Dreamshaper UNet and scheduler
unet = UNet2DConditionModel.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7", 
    subfolder="unet", 
    torch_dtype=torch.float16
)
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "Lykon/dreamshaper-7",
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Load ViT model for image captioning
vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

# Move pipeline and ViT model to GPU or MPS if available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pipe.to(device)
vit_feature_extractor.to(device)
vit_model.to(device)


def generate_image(
    prompt,
    sketch_image,
    steps,
    cfg,
    strength,
    style="None",
    use_caption=True,
    negative_prompt="",
):
    # Preprocess sketch image
    sketch_image = cv2.cvtColor(np.array(sketch_image), cv2.COLOR_RGB2GRAY)
    sketch_image = cv2.Canny(sketch_image, 100, 200)
    sketch_image = Image.fromarray(sketch_image).resize((224, 224))  # Resize for ViT

    # Generate caption using ViT (if enabled)
    caption = ""
    if use_caption:
        inputs = vit_feature_extractor(images=sketch_image, return_tensors="pt").to(device)
        outputs = vit_model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()
        caption = vit_model.config.id2label[predicted_class_idx]

    # Modify prompt based on caption and selected style
    prompt = f"{caption}, {prompt}"
    if style != "None":
        prompt = f"{prompt}, {style}"

    # Generate mask from sketch
    mask_image = Image.new("L", (512, 512), 255)  # White background
    mask_image.paste(sketch_image, (0, 0), sketch_image)  # Paste sketch as black

    # Inpaint using Dreamshaper
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=mask_image,
        mask_image=mask_image,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=strength,
    ).images[0]

    return image


# Flask API
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt")
    sketch_image_data = request.json.get("sketch_image")
    steps = request.json.get("steps")
    cfg = request.json.get("cfg")
    strength = request.json.get("strength")
    style = request.json.get("style")
    use_caption = request.json.get("use_caption", True)
    negative_prompt = request.json.get("negative_prompt", "")

    try:
        sketch_image_bytes = base64.b64decode(sketch_image_data)
        sketch_image = Image.open(io.BytesIO(sketch_image_bytes))
    except Exception as e:
        return jsonify({"error": "Invalid base64 sketch image data" + str(e)})

    image = generate_image(
        prompt, sketch_image, steps, cfg, strength, style, use_caption, negative_prompt
    )

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"image": image_data})


if __name__ == "__main__":
    app.run(debug=True)
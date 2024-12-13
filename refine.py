import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

task = 1

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

init_image = Image.open(f"./output/inpainting_images/{task}/inpainting_image.png").convert("RGB")

prompt = "A cat on the right and a dog on the left."
image = pipe(prompt, image=init_image).images[0]

image.save(f"./output/inpainting_images/{task}/refined_image.png")
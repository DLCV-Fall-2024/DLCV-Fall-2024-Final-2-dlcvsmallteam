import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=int, default=1, help='Task number')

args = parser.parse_args()

task = args.task

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

images = glob.glob(f"./3/*.png")

for i, image in enumerate(images):
    init_image = Image.open(image).convert("RGB")
    prompt = 'A cat wearing wearable glasses in a watercolor style'
    image = pipe(prompt, image=init_image).images[0]

    image.save(f"./3/refined_imag{i}.png")
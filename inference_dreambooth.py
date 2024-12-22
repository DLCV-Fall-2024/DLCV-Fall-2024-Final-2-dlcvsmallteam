import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
import os

pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16
).to('cuda')

pipe.load_lora_weights(
    os.path.join(os.getcwd(), 
                f'lora-trained-xl/cat2/checkpoint-500/pytorch_lora_weights.safetensors')
)

image = pipe(
    prompt = 'a sks cat',
).images[0]

image.save('cat.png')

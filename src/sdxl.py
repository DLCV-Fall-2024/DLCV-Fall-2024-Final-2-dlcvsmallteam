from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16
)

pipeline = pipeline.to('cuda')

prompt = "A realistic photograph of a dog and a cat."
image = pipeline(
    prompt=prompt,
    guidance_scale=8.0,
    num_inference_steps=100,
).images[0]

image.save('test.png')
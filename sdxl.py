from diffusers import StableDiffusionXLPipeline, FluxPipeline
import torch

def generate_base_image(
    prompt = 'A photo.',
    image_filename = 'test.png',
    guidance_scale = 5.0,
    num_inference_steps = 100,
):

    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     'stabilityai/stable-diffusion-xl-base-1.0',
    #     torch_dtype=torch.float16
    # )
    
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        variant="fp16", 
        use_safetensors=True
    )

    # pipeline = pipeline.to('cuda')
    pipeline.enable_model_cpu_offload()

    image = pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=512,
        width=512
    ).images[0]

    image.save(image_filename)

if __name__ == '__main__':
    generate_base_image()
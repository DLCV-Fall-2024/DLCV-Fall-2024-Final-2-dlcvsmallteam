from OmniGen import OmniGenPipeline

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  

input_image_path = 'output/inpainting_images/3/refined_image.png'
style_image_path = 'Data/concept_image/watercolor/image_01_03.jpg'

prompt = 'A cat wearing wearable glasses in a watercolor style shown in <img><|image_1|><\img>.' \
    'The cat is the cat in <img><|image_1|></img>, they have the same details.' \
        'The glasses are the glasses in <img><|image_1|></img>, they have the same details.' \
            'The watercolor style of glasses is the style in <img><|image_2|></img>, they have the same style.' 
            
prompt = 'Change the image [A cat wearing wearable glasses] shown in <img><|image_1|></img> into the watercolor style. The watercolor style is shown in <img><|image_2|></img>.'
            
images = pipe(
    prompt=prompt,
    input_images=[input_image_path, style_image_path],
    height=512, 
    width=512,
    guidance_scale=5., 
    img_guidance_scale=5.,
)
images[0].save(f"output/inpainting_images/3/style.png")


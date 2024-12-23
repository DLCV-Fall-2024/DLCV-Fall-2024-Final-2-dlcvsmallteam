from OmniGen import OmniGenPipeline
import argparse
import os

parser = argparse.ArgumentParser(description='Generate images using OmniGen')
parser.add_argument('-t', '--task', type=int, default=3, help='Task number')

args = parser.parse_args()

pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")  

input_images_list = [
    ['Data/concept_image/cat2/00.jpg',
    'Data/concept_image/dog6/00.jpg'],
    ['Data/concept_image/flower_1/0.jpg',
    'Data/concept_image/vase/02.jpg'],
    ['Data/concept_image/dog/00.jpg',
    'Data/concept_image/pet_cat1/jeanie-de-klerk-av2WGfogjqg-unsplash.jpg',
    'Data/concept_image/dog6/00.jpg'],
    ['Data/concept_image/cat2/00.jpg',
    'Data/concept_image/wearable_glasses/0.png',
    'Data/concept_image/watercolor/image_01_01.jpg']
]

prompt_list = [
    "A scene with a cat on the right and a dog on the left, positioned naturally. The cat is the cat in <img><|image_1|></img>, and the dog is the dog in <img><|image_2|></img>.",
    "A flower placed elegantly in a vase, with the flower's petals vibrant and detailed, and the vase showcasing a simple yet classic design against a neutral background." \
        "The flower is the flower in <img><|image_1|></img>, and the vase is the vase in <img><|image_2|></img>.",
    "There are two dogs and a cat near a forest. One of the dog is the dog in <img><|image_1|></img>, the other dog is the dog in <img><|image_3|></img>, and the cat is the cat in <img><|image_2|></img>.",
    "A cat wearing glasses in the style of watercolor painting. The cat is the cat in <img><|image_1|></img>, the glasses are the glasses in <img><|image_2|></img>, and the watercolor painting is the painting in <img><|image_3|></img>."
]

prompt1 = "A flower placed elegantly in a vase on a table." \
    "The flower is the white flower in <img><|image_1|></img>. " \
        "The vase is the tall and slender vase with angular features in <img><|image_2|></img>." \

prompt2 = "Two dogs and a cat near a forest. " \
    "One of the dog is on the left. " \
        "The other dog is on the right. " \
            "The cat is in the middle. " \
                "The cat is the persian cat in <img><|image_2|></img>. " \
                    "The left dog is the shiba inu dog in <img><|image_1|></img>. " \
                        "The right dog is the corgi dog in <img><|image_3|></img>."
                        
prmopt3 = "A cat wearing glasses in the style of watercolor painting. " \
    "The cat is the British shorthair cat in <img><|image_1|></img>. " \
        "The glasses are the tortoiseshell pattern glasses in <img><|image_2|></img>. " \
            "The tortoiseshell pattern should be detailed and vibrant. " \
                "The watercolor painting is the painting in <img><|image_3|></img>." \
                    "The watercolor style should fill the whole image and the cat should not on the border of the image."

prompt_list[1] = prompt1
prompt_list[2] = prompt2
prompt_list[3] = prmopt3

for i in range(len(input_images_list)):
    if args.task != i and args.task != -1:
        continue
    os.makedirs(f'output/base_images_candidates/{i}', exist_ok=True)
    for j in range(50):
        images = pipe(
            prompt=prompt_list[i],
            input_images=input_images_list[i],
            height=512, 
            width=512,
            guidance_scale=2.5, 
            img_guidance_scale=1.6,
            seed=j
        )
        images[0].save(f"output/base_images_candidates/{i}/{j}.png")


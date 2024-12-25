from OmniGen import OmniGenPipeline
import os
import tqdm

def gen_base_imgs(
    task=0,
    num_imgs=10,
):
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

    pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

    prompt0 = "A scene with a cat on the right and a dog on the left, positioned naturally. The cat is the cat in <img><|image_1|></img>, and the dog is the dog in <img><|image_2|></img>."

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

    prompt_list = [prompt0, prompt1, prompt2, prmopt3]

    

    for i in range(4):
        if task != i and task != -1:
            continue
        os.makedirs(f'codalab_output/base_images/{i}', exist_ok=True)
        pbar = tqdm.tqdm(range(num_imgs))
        for j in pbar:
            images = pipe(
                prompt=prompt_list[i],
                input_images=input_images_list[i],
                height=512, 
                width=512,
                guidance_scale=2.5, 
                img_guidance_scale=1.6,
                seed=j
            )
            for k, img in enumerate(images):
                img.save(f'codalab_output/base_images/{i}/{j}.png')

            pbar.set_description(f"Task {i} - Image {j}")
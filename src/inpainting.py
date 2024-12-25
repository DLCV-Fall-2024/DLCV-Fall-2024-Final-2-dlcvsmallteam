import os
import torch
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src.utils.xml_utils import get_tags_from_xml_file, get_counter_from_tags
from src.utils.florence_utils import generate_image_mask

def generate_inpainting_image(
    task=0,
    output_dir='',
    base_image_path='assets/dog_and_cat.png',
    xml_path='./xmls/0.xml',
    image_filename='assets/test.png',
    tag_list=['cat', 'dog'],
    save_submission=False,
    submission_dir='',
    submission_num=0,
):
    
    base_image = Image.open(base_image_path)
    base_image.save(os.path.join(output_dir, 'base_image.png'))
    
    MODEL_ID = 'microsoft/Florence-2-large-ft'
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    concepts = get_tags_from_xml_file(xml_path)
    counters = get_counter_from_tags(concepts)
    
    # print('Concepts: ', concepts)
    # print('Counters: ', counters)
    
    TASK_PROMPT = '<CAPTION_TO_PHRASE_GROUNDING>'
    
    img_cnt = 0
    
    for i, (depth, tags) in enumerate(concepts.items()):
        
        if i == 0:
            continue
        
        unique_tags = list(set(tags))
        unique_tags.sort()
        
        # print(f'Layer: {i}, Unique tags: {unique_tags}')
        
        if i == 1:
            image = Image.open(base_image_path)
        else:
            image = Image.open(os.path.join(output_dir, image_filename))
        
        for idx, tag in enumerate(unique_tags):
        
            # print(f'i: {i}, idx: {idx}, tag: {tag}')
            
            prompt = TASK_PROMPT + tag
            inputs = processor(text=prompt, images=image, return_tensors='pt')
            
            generated_ids = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                pixel_values=inputs['pixel_values'].cuda(),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3
            )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            parsed_result = processor.post_process_generation(
                generated_text,
                task=TASK_PROMPT,
                image_size=(image.width, image.height)
            )
            
            boxes = parsed_result[TASK_PROMPT]['bboxes']
            
            for k, box in enumerate(boxes):
                image_mask = generate_image_mask(image, bbox=box)
                # image_mask = image_mask.filter(ImageFilter.GaussianBlur(5))
                image_mask.save(os.path.join(output_dir, f'mask_{tag}_{k}.png'))
            
            number_same_tag = counters[depth][tag]
            
            for k in range(number_same_tag):
                
                concept_name = tag_list[img_cnt]
                
                pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                    'stabilityai/stable-diffusion-xl-base-1.0',
                    torch_dtype=torch.float16
                ).to('cuda')
                
                pipeline.load_lora_weights(os.path.join(os.getcwd(),
                    f'lora-trained-xl/{concept_name}/checkpoint-500/pytorch_lora_weights.safetensors'))
                
                if img_cnt == 0:
                    origin_image = Image.open(base_image_path).convert('RGB')
                else:
                    origin_image = Image.open(os.path.join(output_dir, image_filename)).convert('RGB')
                    
                mask_image = Image.open(os.path.join(output_dir, f'mask_{tag}_{k}.png')).convert('L')
                
                if task == 0:
                    if concept_name == 'dog6':
                        prompt = f'The sitting cat on the right and the sitting sks dog on the left.'
                    elif concept_name == 'cat2':
                        prompt = f'The sitting sks cat on the right and the sitting dog on the left.'
                elif task == 1:
                    if concept_name == 'vase':
                        prompt = f'a sks vase.'
                    elif concept_name == 'flower_1':
                        prompt = f'a sks flower in the vase.'
                elif task == 2:
                    if concept_name == 'dog6':
                        prompt = "Two dogs and a cat near a forest. " \
                                "One of the sks dog is on the left. " \
                                    "The other dog is on the right. " \
                                        "The cat is in the middle. "
                    elif concept_name == 'dog':
                        prompt = "Two dogs and a cat near a forest. " \
                                "One of the dog is on the left. " \
                                    "The other sks dog is on the right. " \
                                        "The cat is in the middle. "
                    elif concept_name == 'pet_cat1':
                        prompt = "Two dogs and a cat near a forest. " \
                                "One of the dog is on the left. " \
                                    "The other dog is on the right. " \
                                        "The sks pet cat is in the middle. "
                elif task == 3:
                    prompt = f'A sks {tag}.'
                
                strength_list = [0.3, 0.3, 0.1, 0.1]

                if concept_name == 'vase':
                    strength_list = [0.3, 0.9999, 0.1, 0.1]
                else:
                    strength_list = [0.3, 0.3, 0.1, 0.1]

                result = pipeline(
                    prompt=prompt,
                    image=origin_image,
                    mask_image=mask_image,
                    strength=strength_list[task],
                    num_inference_steps=50,
                    # width=512,
                    # height=512
                ).images[0]
                
                result.save(os.path.join(output_dir, image_filename))
                result.save(os.path.join(output_dir, f'{concept_name}_{img_cnt}.png'))
                
                img_cnt += 1
                
    if save_submission:
        refine_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-refiner-1.0',
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True
        ).to('cuda')
        
        prompt_list = [
            'A cat on the right and a dog on the left.',
            'A flower in a vase.',
            'A dog, a pet cat and a dog near a forest.',
            'A cat wearing wearable glasses in a watercolor style'
        ]
        
        refined_image = refine_pipeline(
            image=result,
            prompt=prompt_list[task],
            # original_size=(512,512),
            # target_size=(512,512)
        ).images[0]
        
        refined_image = refined_image.resize((512, 512))

        refined_image.save(os.path.join(submission_dir, f'{submission_num}.png'))        
    
if __name__ == '__main__':
    generate_inpainting_image()
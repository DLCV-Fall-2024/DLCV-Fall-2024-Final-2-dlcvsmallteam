from inpainting import generate_inpainting_image
import os
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=int, default=0, help='Task number')
    parser.add_argument('-s', '--submission_type', action='store_true', help='Generate Submission type')
    
    args = parser.parse_args()
    
    task = args.task
    
    tag_list_list = [
        ['cat2', 'dog6'],
        ['flower_1', 'vase'],
        ['pet_cat1', 'dog', 'dog6', ],
        ['cat2', 'wearable_glasses', 'watercolor']
    ]
    
    output_dir = f'./output/inpainting_images/{task}'
    xml_path = f'./xmls/{task}.xml'
    # base_image_prompt_path = os.path.join('./llm_base_image_prompts', f'{task}.txt')
    
    # with open(base_image_prompt_path, 'r') as file:
    #     base_image_prompt = file.read()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # generate_base_image(
    #     prompt = base_image_prompt,
    #     image_filename = os.path.join(output_dir, 'base_image.png'),
    # )
    
    base_image_path = f'./output/base_images/{task}/0.png'
    
    if not args.submission_type:
        generate_inpainting_image(
            task=task,
            output_dir=output_dir,
            base_image_path = base_image_path,
            xml_path = xml_path,
            tag_list = tag_list_list[task],
            image_filename= 'inpainting_image.png'
        )
    else:
        for i in range(10):
            generate_inpainting_image(
                task=task,
                output_dir=output_dir,
                base_image_path = base_image_path,
                xml_path = xml_path,
                tag_list = tag_list_list[task],
                image_filename= f'test_{i}.png',
                save_submission=True,
                submission_dir='./output/submission_images',
                submission_num=i
            )

if __name__ == '__main__':
    main()
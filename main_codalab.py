from src.inpainting import generate_inpainting_image
from src.inpainting_ctrl import generate_inpainting_image_ctrl
from src.inpainting_flux import generate_inpainting_image_flux
import os
import argparse
import glob

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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

    os.makedirs(output_dir, exist_ok=True)
    
    
    base_images = glob.glob(f'./output/base_images/{task}/*.png')
    base_images = sorted(base_images)
    
    if not args.submission_type:
        generate_inpainting_image(
            task=task,
            output_dir=output_dir,
            base_image_path = base_images[0],
            xml_path = xml_path,
            tag_list = tag_list_list[task],
            image_filename= 'inpainting_image.png'
        )
    else:
        for j in range(10):
            for i in range(10):
                print(f'Generating submission image {i}')
                generate_inpainting_image(
                    task=task,
                    output_dir=output_dir,
                    base_image_path = base_images[i],
                    xml_path = xml_path,
                    tag_list = tag_list_list[task],
                    image_filename= f'test_{10*j+i}.png',
                    save_submission=True,
                    submission_dir='./output/submission_images_candidates',
                    submission_num=10*j+i
                )

if __name__ == '__main__':
    main()
from src.gen_base_imgs import gen_base_imgs
from src.inpainting import generate_inpainting_image
import os
import argparse
import glob

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=int, default=0, help='Task number')

    args = parser.parse_args()
    
    task = args.task
    
    tag_list_list = [
        ['cat2', 'dog6'],
        ['flower_1', 'vase'],
        ['pet_cat1', 'dog', 'dog6', ],
        ['cat2', 'wearable_glasses', 'watercolor']
    ]
    
    output_dir = f'./codalab_output/inpainting_images/{task}'
    submission_candidates_dir = f'./codalab_output/submission_images_candidates/{task}'
    submission_dir = f'./codalab_output/submission_images/{task}'
    xml_path = f'./xmls/{task}.xml'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submission_candidates_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)
    
    NUM_IMAGES = 100

    gen_base_imgs(task=task, num_imgs=NUM_IMAGES)

    base_images = glob.glob(f'./codalab_output/base_images/{task}/*.png')
    base_images = sorted(base_images)

    gen_base_imgs(task=task, num_imgs=NUM_IMAGES)

    for i in range(NUM_IMAGES):
        print(f'Generating submission image {i}')
        generate_inpainting_image(
            task=task,
            output_dir=output_dir,
            base_image_path = base_images[i],
            xml_path = xml_path,
            tag_list = tag_list_list[task],
            image_filename= f'test_{i}.png',
            save_submission=True,
            submission_dir=submission_candidates_dir,
            submission_num=i
        )


if __name__ == '__main__':
    main()
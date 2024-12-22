import shutil
import json
import glob
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description='Select base images for inpainting')

    parser.add_argument('-t', '--task', type=int, default=3, help='Task number')

    args = parser.parse_args()

    with open('./assets/select_base_images.json', 'r') as f:
        selected_images_idx = json.load(f)

    print(selected_images_idx)

    for i in range(4):

        if args.task != i and args.task != -1:
            continue

        os.makedirs(os.path.join('./output/base_images', str(i)), exist_ok=True)

        for j in range(10):

            source_image = os.path.join('./output/base_images_candidates', str(i), f'{selected_images_idx[str(i)][j]}.png')

            shutil.copy2(source_image, os.path.join('./output/base_images', str(i), f'{str(j)}.png'))


if __name__ == '__main__':
    main()
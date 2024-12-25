import argparse
import os

from src.selection import select_images
from src.selection_best import select_best_images

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=int, default=0, help='Task number')
    parser.add_argument('-b', '--best', action='store_true', help='Select best images')
    args = parser.parse_args()

    task = args.task
    image_dir = os.path.join('codalab_output/submission_images_candidates', str(task))
    json_path = 'Data/prompts.json'
    save_dir = os.path.join('codalab_output/submission_images', str(task))
    
    if not args.best:
        select_images(task, image_dir, json_path, save_dir)
    else:
        select_best_images(task, image_dir, json_path, save_dir)

if __name__ == '__main__':
    main()
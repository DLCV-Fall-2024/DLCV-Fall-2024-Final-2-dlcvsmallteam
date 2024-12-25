import argparse
import os

from src.selection import select_best_images

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=int, default=0, help='Task number')
    args = parser.parse_args()

    task = args.task
    image_dir = os.path.join('codalab_output/submission_images_candidates', str(task))
    json_path = 'Data/prompts.json'
    save_dir = os.path.join('codalab_output/submission_images', str(task))
    
    select_best_images(task, image_dir, json_path, save_dir)

if __name__ == '__main__':
    main()
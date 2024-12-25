from src.gen_base_imgs import gen_base_imgs
import os
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=int, default=0, help='Task number')

    args = parser.parse_args()
    
    task = args.task
    
    NUM_IMAGES = 100

    gen_base_imgs(task=task, num_imgs=NUM_IMAGES)


if __name__ == '__main__':
    main()
import glob
import json
import os

from PIL import Image

from src.clip_score import calculate_clip_image_scores_folder, calculate_clip_text_scores_folder

def select_best_images(task, image_dir, json_path, save_dir):
    
    with open(json_path, 'r') as file:
        prompts = json.load(file)

    key, value = list(prompts.items())[task]

    image_path_list = glob.glob(os.path.join(image_dir, '*.png'))
    image_path_list = sorted(image_path_list)

    all_input_dirs = [
        ['Data/concept_image/cat2', 'Data/concept_image/dog6'],
        ['Data/concept_image/flower_1', 'Data/concept_image/vase'],
        ['Data/concept_image/dog', 'Data/concept_image/pet_cat1', 'Data/concept_image/dog6'],
        ['Data/concept_image/cat2', 'Data/concept_image/wearable_glasses', 'Data/concept_image/watercolor']
    ]

    image_scores = calculate_clip_image_scores_folder(
        image_path_list,
        all_input_dirs[task],
    )

    text_scores = calculate_clip_text_scores_folder(
        image_path_list,
        value['prompt_4_clip_eval'],
    )

    total_score = [image_score + 2 * text_score for image_score, text_score in zip(image_scores, text_scores)]

    sorted_images_path = sorted(zip(image_path_list, total_score), key=lambda x: x[1], reverse=True)

    selected_images_path = sorted_images_path[0]

    selected_images = Image.open(selected_images_path)

    for i in range(10):
        selected_images.save(os.path.join(save_dir, f'{i}.png'))

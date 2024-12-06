import json
import argparse
import os
from clip_image import calculate_clip_image_scores_folder
from clip_text import calculate_clip_text_scores_folder

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process some JSON data.')
    parser.add_argument('--json_path', type=str, help='Path to the JSON file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory of saved output')
    parser.add_argument('-s', '--setting', type=int, default=0, help='Setting of the evaluation')
    # Parse the arguments
    args = parser.parse_args()
    json_path = args.json_path
    output_dir = args.output_dir
    
    assert args.setting in [0, 1, 2, 3], "Setting must be 0, 1, 2, or 3"
    
    # Assuming the JSON is saved in a file named 'data.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    all_input_dirs = [
        ['Data/concept_image/cat2', 'Data/concept_image/dog6'],
        ['Data/concept_image/flower_1', 'Data/concept_image/vase'],
        ['Data/concept_image/dog', 'Data/concept_image/pet_cat1', 'Data/concept_image/dog6'],
        ['Data/concept_image/cat2', 'Data/concept_image/wearable_glasses', 'Data/concept_image/watercolor']
    ]

    pass_count = 0
    count = 0
    print("\n===============================================start evaluation================================================\n")

    # Iterate through the data and print the prompt_4_clip_eval
    key, value = list(data.items())[args.setting]
    
    input_folder_path = all_input_dirs[args.setting]
    output_folder_path = os.path.join(output_dir, str(args.setting))

    src = value['src_image']
    
    clip_eval = value['prompt_4_clip_eval']

    print(f"Image source: \"{src}\", text prompt: {clip_eval}")

    image_scores = calculate_clip_image_scores_folder(output_folder_path, input_folder_path)
    text_scores = calculate_clip_text_scores_folder(output_folder_path, clip_eval)
    
    # total_score = image_scores + 2.5 * text_scores
    image_score = sum(image_scores) / len(image_scores)
    text_scores = sum(text_scores) / len(text_scores)

    print(f"CLIP Image Score: {image_score:.2f}")
    print(f"CLIP Text Score: {text_scores:.2f}")



    